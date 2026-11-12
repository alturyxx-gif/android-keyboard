#include <chrono>
#include <vector>
#include <string>
#include <cstring>
#include <cmath>
using std::isnan;
// Tests the personalisation pipeline end to end against the real model:
// train a LoRA adapter (as the TrainingWorker does), then load the base model with the
// adapter applied at runtime (as LanguageModel does) and check the learned text became
// more likely. Also checks a corrupt adapter falls back to the plain base model.
#include "../org_futo_inputmethod_latin_xlm_AdapterTrainer.cpp"
#include <cstdio>
#include <cmath>
#include <fstream>

// From test_scoring-style helper compiled in a separate TU
float host_score_continuation(const char *model, const char *adapter, float scale, const char *prefix, const char *cont);

using namespace latinime;

static int failures = 0;
#define CHECK(cond, ...) do { if(!(cond)) { failures++; printf("  FAIL: " __VA_ARGS__); printf("\n"); } else { printf("  ok: " __VA_ARGS__); printf("\n"); } } while(0)

static void onLoss(void *, float loss) { static int i = 0; if((i++ % 16) == 0) printf("    loss %.3f\n", loss); }
static void onProgress(void *, float) {}

int main(int argc, char **argv) {
    if(argc < 2) { printf("usage: %s model.gguf\n", argv[0]); return 2; }
    const char *model = argv[1];
    const char *adapter = "/tmp/xlm_host_test.lora";
    remove(adapter);

    // A made-up name + phrase that the base model can't know
    const char *prefix = "Tomorrow I am meeting";
    const char *cont = "Zarnaab at the cafe";
    float base = host_score_continuation(model, "", 0.0f, prefix, cont);
    printf("== Base model: log P('%s' | '%s') = %.3f\n", cont, prefix, base);

    auto *st = new AdapterTrainerState();
    st->baseModelPath = model;
    st->loraCachePath = adapter;
    st->outputModelPath = "";
    st->outputScale = 1.0f;
    if(!st->Initialize()) { printf("trainer init failed\n"); return 1; }
    st->params.common.callbacks.userdata = st;
    st->params.common.callbacks.loss = onLoss;
    st->params.common.callbacks.progress = onProgress;

    // Realistic, varied "typing history": everyday messages with a few personal facts
    // (a name the base model can't know, a favourite place) mixed in.
    std::vector<std::string> lines;
    const char *people[] = {"Zarnaab", "Ahmed", "my sister", "the team", "mom"};
    const char *places[] = {"the cafe", "the office", "the gym", "Lahore", "the mall"};
    const char *templates[] = {
        "Tomorrow I am meeting %s at %s ", "Can you ask %s about %s ", "I will see %s near %s later ",
        "%s said %s was really busy today ", "Let's go to %s with %s ", "I just got back from %s, %s says hi ",
        "Are you coming to %s with %s ", "Remind me to call %s before %s closes ",
    };
    int k = 0;
    for(auto t : templates) for(auto p : people) for(auto pl : places) {
        char buf[256];
        bool personFirst = (k++ % 2) == 0;
        if(strstr(t, "%s at %s") || strstr(t, "%s about %s") || strstr(t, "%s near %s") || strstr(t, "call %s"))
            snprintf(buf, sizeof buf, t, p, pl);
        else if(strstr(t, "%s said %s")) snprintf(buf, sizeof buf, t, p, pl);
        else snprintf(buf, sizeof buf, t, pl, p);
        (void)personFirst;
        lines.push_back(buf);
    }
    const char *misc[] = {
        "I am running a bit late ", "Did you finish the report ", "What time is the meeting ",
        "Thanks for the help yesterday ", "Can we move it to Friday ", "I'll send the photos tonight ",
        "Don't forget the groceries ", "How was the trip ", "Let me check and get back to you ",
        "Sounds good to me ", "I'm at home now ", "Call me when you are free ",
    };
    for(int r = 0; r < 4; r++) for(auto m : misc) lines.push_back(m);
    for(auto &l : lines) st->AddTrainingExample(l);
    int steps = (int)lines.size() / 4;  // one pass, as TrainingPlan.steps()
    printf("== %zu training lines, %d steps\n", lines.size(), steps);

    st->SetPlan(steps, /*ctx*/ 64, /*batch*/ 4, /*threads*/ 4);
    printf("== Training LoRA adapter\n");
    auto t0 = std::chrono::steady_clock::now();
    int rc = st->Train();
    CHECK(rc == 0, "training succeeded");
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("  training took %.0f s (%.2f s/step at ctx 64, 4 threads)\n", secs, secs / steps);
    std::ifstream f(adapter, std::ios::binary | std::ios::ate);
    CHECK(f.good() && f.tellg() > 0, "adapter file written (%lld bytes)", (long long)(f.good() ? (long long)f.tellg() : 0));
    delete st;

    float full = host_score_continuation(model, adapter, 1.0f, prefix, cont);
    float dflt = host_score_continuation(model, adapter, 0.75f, prefix, cont);
    float half = host_score_continuation(model, adapter, 0.5f, prefix, cont);
    float off  = host_score_continuation(model, adapter, 0.0f, prefix, cont);
    printf("== Learned phrase: strength 1.0 = %.3f, 0.75 = %.3f, 0.5 = %.3f, 0.0 = %.3f (base %.3f)\n", full, dflt, half, off, base);
    CHECK(dflt > base + 1.0f, "adapter (default strength 0.75) makes the learned phrase much more likely");
    CHECK(half > base && half < full, "strength 0.5 is in between");
    CHECK(std::fabs(off - base) < 1e-3, "strength 0 is exactly the base model");

    float base2 = host_score_continuation(model, "", 0.0f, prefix, cont);
    CHECK(std::fabs(base2 - base) < 1e-3, "base model file unchanged");

    // Same everyday-English check the TrainingWorker uses (GeneralEnglishCheck), avg per line
    const char *general[][2] = {
        {"I am going to the","store"}, {"Can you send me the","link"}, {"What time does the","meeting"},
        {"Thank you so much for your","help"}, {"I will be there in ten","minutes"}, {"Let me know if you","need"},
        {"Do you want to grab","lunch"}, {"I think we should","go"}, {"How was your","day"},
        {"See you","tomorrow"}, {"Sorry I missed your","call"}, {"Happy","birthday"},
        {"I don't know what to","do"}, {"It was really nice to","meet"}, {"Could you please","send"},
        {"I'm on my","way"}, {"Have a good","night"}, {"That sounds like a good","idea"},
    };
    double g0 = 0, g1 = 0; int ng = 0;
    for(auto &g : general) {
        g0 += host_score_continuation(model, "", 0.0f, g[0], g[1]);
        g1 += host_score_continuation(model, adapter, 0.75f, g[0], g[1]);
        ng++;
    }
    g0 /= ng; g1 /= ng;
    printf("== Everyday English (avg log-prob per word): base %.3f, with adapter %.3f (drop %.3f, limit 0.35)\n", g0, g1, g0 - g1);
    CHECK(g1 >= g0 - 0.35, "everyday English not made noticeably worse (TrainingWorker would keep this adapter)");

    // Corrupt adapter -> falls back to base model instead of failing
    { std::ofstream bad("/tmp/xlm_host_bad.lora", std::ios::binary); bad << "not a lora"; }
    float corrupt = host_score_continuation(model, "/tmp/xlm_host_bad.lora", 1.0f, prefix, cont);
    CHECK(std::isfinite(corrupt) && std::fabs(corrupt - base) < 1e-3, "corrupt adapter falls back to base model");

    printf("\n%s (%d failures)\n", failures == 0 ? "ALL PASSED" : "SOME FAILED", failures);
    return failures == 0 ? 0 : 1;
}
