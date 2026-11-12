#include <vector>
#include <string>
#include <cstring>
#include <cmath>
using std::isnan;
// Tests ScoreContinuations (used by live grammar correction) against the real model.
#include "../org_futo_inputmethod_latin_xlm_LanguageModel.cpp"
#include <cstdio>

using namespace latinime;

static int failures = 0;
#define CHECK(cond, ...) do { if(!(cond)) { failures++; printf("  FAIL: " __VA_ARGS__); printf("\n"); } else { printf("  ok: " __VA_ARGS__); printf("\n"); } } while(0)

struct Case { const char *prefix; const char *wrong; const char *right; const char *rest; };

int main(int argc, char **argv) {
    if(argc < 2) { printf("usage: %s model.gguf\n", argv[0]); return 2; }
    auto *state = new LanguageModelState();
    if(!state->Initialize(argv[1])) { printf("model load failed\n"); return 1; }

    printf("== Basic scoring\n");
    auto s = ScoreContinuations(state, "I really like", {"their car", "there car"});
    printf("  their car=%.3f  there car=%.3f\n", s[0], s[1]);
    CHECK(s[0] > s[1], "'their car' scores higher than 'there car'");

    auto s2 = ScoreContinuations(state, "", {"Hello how are you", "Hello how are"});
    CHECK(std::isfinite(s2[0]) && std::isfinite(s2[1]), "empty prefix works");
    CHECK(s2[0] < s2[1], "longer continuation has lower total log-prob");

    printf("== Determinism / cache consistency\n");
    auto a1 = ScoreContinuations(state, "We went to the", {"park yesterday"});
    auto noise = ScoreContinuations(state, "Something completely different here", {"banana"});
    auto a2 = ScoreContinuations(state, "We went to the", {"park yesterday"});
    printf("  first=%.5f second=%.5f\n", a1[0], a2[0]);
    CHECK(std::fabs(a1[0] - a2[0]) < 1e-3, "same input gives same score after other calls (KV cache reuse is correct)");

    // Suggestions must still work after scoring touched the shared KV cache
    auto before = state->PredictNextWord("I am going to the", {});
    ScoreContinuations(state, "I am going to", {"the store", "a party now"});
    auto after = state->PredictNextWord("I am going to the", {});
    bool same = before.size() == after.size();
    for(size_t i = 0; same && i < before.size(); i++) same = before[i].second == after[i].second;
    printf("  next word after 'I am going to the': ");
    for(auto &p : after) printf("[%s %.3f] ", p.second.c_str(), p.first);
    printf("\n");
    CHECK(same, "next-word predictions identical before/after scoring");

    printf("== Grammar margins (wrong -> right should be positive and large)\n");
    // These mirror GrammarChecker: margin = score(prefix, right+rest) - score(prefix, wrong+rest)
    Case cases[] = {
        {"I really like", "there", "their", " car, it is fast"},
        {"I think", "its", "it's", " going to rain today"},
        {"The dog wagged", "it's", "its", " tail happily"},
        {"I hope", "your", "you're", " feeling better"},
        {"Is this", "you're", "your", " phone on the table"},
        {"She is taller", "then", "than", " her brother"},
        {"Every morning he", "go", "goes", " to the gym"},
        {"They", "was", "were", " at the party last night"},
        {"I would like", "a", "an", " apple please"},
        {"It is", "an", "a", " good idea"},
        {"I could", "of", "have", " done it better"},
        {"I have", "went", "gone", " there before"},
        {"Yesterday I", "seen", "saw", " a movie"},
        {"They left", "there", "their", " keys at home"},
        {"Over", "their", "there", " is a big tree"},
    };
    int n = 0, autoApply = 0, underline = 0, missed = 0;
    std::vector<float> rightToWrong;
    for(auto &c : cases) {
        std::string w = std::string(c.wrong) + c.rest, r = std::string(c.right) + c.rest;
        auto sc = ScoreContinuations(state, c.prefix, {w, r});
        float margin = sc[1] - sc[0];
        rightToWrong.push_back(-margin);
        n++;
        const char *verdict = margin >= 4.0f ? "AUTO" : margin >= 2.0f ? "UNDERLINE" : "missed";
        if(margin >= 4.0f) autoApply++; else if(margin >= 2.0f) underline++; else missed++;
        printf("  %-18s %-7s -> %-7s margin=%6.2f  %s\n", c.prefix, c.wrong, c.right, margin, verdict);
    }
    printf("  summary: %d auto-applied, %d underlined, %d missed (of %d)\n", autoApply, underline, missed, n);
    CHECK(missed <= n / 3, "most real mistakes are caught");

    printf("== False positives (correct text -> wrong alternative must stay below thresholds)\n");
    int falsePositives = 0;
    for(size_t i = 0; i < rightToWrong.size(); i++) {
        if(rightToWrong[i] >= 2.0f) { falsePositives++; printf("  FALSE POSITIVE: case %zu margin %.2f\n", i, rightToWrong[i]); }
    }
    CHECK(falsePositives == 0, "no correct sentence would be changed");

    printf("== False positives on correct everyday sentences\n");
    // Each correct sentence is split at a confusable word; every alternative must NOT beat the
    // original by the underline threshold (2.0), otherwise the checker would "fix" correct text.
    struct Correct { const char *prefix; const char *word; std::vector<const char*> alts; const char *rest; };
    std::vector<Correct> correct = {
        {"I want", "to", {"too"}, " go home now"},
        {"That is way", "too", {"to"}, " expensive for me"},
        {"Put it over", "there", {"their", "they're"}, " on the table"},
        {"The kids forgot", "their", {"there", "they're"}, " lunch at home"},
        {"I think", "they're", {"their", "there"}, " coming tomorrow"},
        {"", "It's", {"Its"}, " a really nice day"},
        {"The company changed", "its", {"it's"}, " name last year"},
        {"Thank you,", "you're", {"your"}, " very kind"},
        {"Where did you park", "your", {"you're"}, " car"},
        {"We ate and", "then", {"than"}, " we went for a walk"},
        {"This one is better", "than", {"then"}, " the old one"},
        {"She", "goes", {"go"}, " to school by bus"},
        {"They", "go", {"goes"}, " to school by bus"},
        {"We", "were", {"was"}, " late again"},
        {"He", "was", {"were"}, " late again"},
        {"I need", "a", {"an"}, " new phone"},
        {"It took", "an", {"a"}, " hour to get there"},
        {"I have", "seen", {"saw"}, " that movie twice"},
        {"I", "saw", {"seen"}, " her yesterday"},
        {"She has", "gone", {"went"}, " to the shop"},
        {"I could", "have", {"of"}, " helped you"},
        {"Most", "of", {"have"}, " my friends live nearby"},
        {"Did you", "lose", {"loose"}, " your keys again"},
        {"I don't know", "whether", {"weather"}, " I can come"},
        {"The", "weather", {"whether"}, " is lovely today"},
        {"This will", "affect", {"effect"}, " everyone"},
        {"Please", "accept", {"except"}, " my apology"},
        {"It was", "quite", {"quiet"}, " good actually"},
        {"Can I have a", "piece", {"peace"}, " of cake"},
        {"I", "do", {"does"}, " not like it"},
        {"He", "does", {"do"}, " not like it"},
    };
    int fp = 0, total = 0;
    for(auto &c : correct) {
        std::vector<std::string> conts = { std::string(c.word) + c.rest };
        for(auto a : c.alts) conts.push_back(std::string(a) + c.rest);
        auto sc = ScoreContinuations(state, c.prefix, conts);
        for(size_t j = 1; j < sc.size(); j++) {
            total++;
            float margin = sc[j] - sc[0];
            bool bad = margin >= 2.0f;
            if(bad) fp++;
            printf("  %-22s %-8s -> %-8s margin=%6.2f %s\n", c.prefix, c.word, c.alts[j-1], margin, bad ? (margin >= 4.0f ? "WOULD AUTO-CHANGE" : "would underline") : "");
        }
    }
    printf("  %d of %d alternatives would wrongly trigger\n", fp, total);
    CHECK(fp == 0, "no false positives on correct sentences");

    delete state;
    printf("\n%s (%d failures)\n", failures == 0 ? "ALL PASSED" : "SOME FAILED", failures);
    return failures == 0 ? 0 : 1;
}
