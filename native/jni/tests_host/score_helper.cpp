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
