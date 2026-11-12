#include <cstring>
#include <cmath>
using std::isnan;
// Separate translation unit: loads a LanguageModelState (with optional adapter) and scores.
#include "../org_futo_inputmethod_latin_xlm_LanguageModel.cpp"

float host_score_continuation(const char *model, const char *adapter, float scale, const char *prefix, const char *cont) {
    auto *state = new LanguageModelState();
    if(!state->Initialize(model, adapter, scale, 4)) { delete state; return NAN; }
    auto s = ScoreContinuations(state, prefix, {cont});
    delete state;
    return s[0];
}

// Scores many (prefix, continuation) pairs with one loaded model; returns the average.
double host_score_average(const char *model, const char *adapter, float scale,
                          const std::vector<std::pair<std::string, std::string>> &pairs) {
    auto *state = new LanguageModelState();
    if(!state->Initialize(model, adapter, scale, 4)) { delete state; return NAN; }
    double s = 0; int n = 0;
    for(auto &p : pairs) {
        auto r = ScoreContinuations(state, p.first, {p.second});
        if(std::isfinite(r[0])) { s += r[0]; n++; }
    }
    delete state;
    return n ? s / n : NAN;
}
