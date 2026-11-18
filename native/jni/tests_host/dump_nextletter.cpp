// Tap plan: writes the typing model's next-letter probabilities for many (sentence, prefix)
// pairs, for replaying recorded taps on a PC. Input lines: "sentence<TAB>prefix".
// Output lines: 27 floats (a..z, then "word finished").
#include <cstring>
#include <cmath>
using std::isnan;
#include "../org_futo_inputmethod_latin_xlm_LanguageModel.cpp"
#include <cstdio>
#include <iostream>

int main(int argc, char **argv) {
    auto *state = new LanguageModelState();
    int threads = argc > 2 ? atoi(argv[2]) : 4;
    if(!state->Initialize(argv[1], "", 0.0f, threads)) { fprintf(stderr, "load failed\n"); return 1; }
    std::string line;
    while(std::getline(std::cin, line)) {
        auto tab = line.find('\t');
        std::string ctx = tab == std::string::npos ? "" : line.substr(0, tab);
        std::string prefix = tab == std::string::npos ? line : line.substr(tab + 1);
        auto p = NextLetterProbs(state, ctx, prefix);
        if(p.size() != 27) { printf("-\n"); continue; }
        for(int i = 0; i < 27; i++) printf(i ? " %.5f" : "%.5f", p[i]);
        printf("\n");
        fflush(stdout);
    }
}
