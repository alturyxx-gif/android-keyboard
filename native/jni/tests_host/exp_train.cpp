// Experiment driver: train an adapter from a lines file, then report held-out personal and
// general-English scores at several adapter strengths. Not part of the test suite.
#include <cstring>
#include <cmath>
using std::isnan;
#include "../org_futo_inputmethod_latin_xlm_AdapterTrainer.cpp"
#include <cstdio>
#include <fstream>
#include <chrono>
float host_score_continuation(const char *model, const char *adapter, float scale, const char *prefix, const char *cont);
using namespace latinime;
static void onLoss(void *, float loss) { static int i = 0; if((i++ % 20) == 0) { printf("    loss %.3f\n", loss); fflush(stdout);} }
static void onProgress(void *, float) {}
static std::vector<std::pair<std::string,std::string>> readPairs(const char *f) {
    std::vector<std::pair<std::string,std::string>> v; std::ifstream in(f); std::string l;
    while(std::getline(in, l)) { auto p = l.find('|'); if(p != std::string::npos) v.push_back({l.substr(0,p), l.substr(p+1)}); }
    return v;
}
double host_score_average(const char *model, const char *adapter, float scale, const std::vector<std::pair<std::string, std::string>> &pairs);
static double avg(const char *model, const char *adapter, float scale, const std::vector<std::pair<std::string,std::string>> &v) {
    return host_score_average(model, adapter, scale, v);
}
int main(int argc, char **argv) {
    // model train.txt personal_eval.txt general_eval.txt lr stepsCap ctx adapter_out
    const char *model = argv[1]; float lr = atof(argv[5]); float epochs = atof(argv[6]); int cap = 300; int ctx = atoi(argv[7]); const char *adapter = argv[8];
    auto pers = readPairs(argv[3]); auto gen = readPairs(argv[4]);
    remove(adapter);
    auto *st = new AdapterTrainerState();
    st->baseModelPath = model; st->loraCachePath = adapter; st->outputModelPath = ""; st->outputScale = 1.0f;
    if(!st->Initialize()) return 1;
    st->params.common.callbacks.userdata = st; st->params.common.callbacks.loss = onLoss; st->params.common.callbacks.progress = onProgress;
    std::ifstream in(argv[2]); std::string l; int n = 0;
    while(std::getline(in, l)) { if(!l.empty()) { st->AddTrainingExample(l); n++; } }
    st->SetPlan(cap, ctx, 4, 4, epochs);
    st->params.common.adam_alpha = lr;
    printf("== %d lines, epochs %.1f, steps %d, lr %g, ctx %d\n", n, epochs, st->params.common.adam_n_iter, lr, ctx); fflush(stdout);
    auto t0 = std::chrono::steady_clock::now();
    int rc = st->Train();
    double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    printf("== train rc %d, %.0f s\n", rc, secs);
    delete st;
    double p0 = avg(model, "", 0, pers), g0 = avg(model, "", 0, gen);
    printf("base: personal %.3f general %.3f\n", p0, g0);
    for(float s : {1.0f, 0.75f, 0.5f, 0.25f}) {
        double p = avg(model, adapter, s, pers), g = avg(model, adapter, s, gen);
        printf("scale %.2f: personal %+.3f  general %+.3f\n", s, p - p0, g - g0); fflush(stdout);
    }
    return 0;
}
