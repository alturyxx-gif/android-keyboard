// Tap plan Stage 3.1: does the typing model predict the next typed letter sensibly, and fast?
#include <cstring>
#include <cmath>
using std::isnan;
#include "../org_futo_inputmethod_latin_xlm_LanguageModel.cpp"
#include <chrono>
#include <cstdio>
#include <algorithm>

static int failures = 0;

static std::string top(const std::vector<float> &p, int n) {
    std::vector<int> idx(26); for(int i = 0; i < 26; i++) idx[i] = i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return p[a] > p[b]; });
    std::string s;
    for(int i = 0; i < n; i++) { char buf[32]; snprintf(buf, sizeof buf, "%c %.2f  ", 'a' + idx[i], p[idx[i]]); s += buf; }
    return s;
}

int main(int argc, char **argv) {
    auto *state = new LanguageModelState();
    if(!state->Initialize(argv[1], "", 0.0f, 4)) { printf("load failed\n"); return 1; }
    struct Case { const char *ctx; const char *prefix; char expect; };
    Case cases[] = {
        {"I'm going to the", "sto", 'r'},
        {"Can you send me the", "lin", 'k'},
        {"Thank you so much for your", "", 'h'},
        {"", "th", 'e'},
        {"See you", "tomo", 'r'},
        {"What time is the", "meet", 'i'},
        {"I need to go to the", "doct", 'o'},
        {"Did you", "ge", 't'},
        {"Let me know if you", "ne", 'e'},
        {"How was your", "", 'd'},
    };
    int ok = 0;
    for(auto &c : cases) {
        auto p = NextLetterProbs(state, c.ctx, c.prefix);
        int best = std::max_element(p.begin(), p.end()) - p.begin();
        bool good = ('a' + best) == c.expect;
        ok += good;
        printf("%-30s +%-6s -> %s %s\n", c.ctx, c.prefix, top(p, 4).c_str(), good ? "" : "(expected first)");
    }
    printf("== %d of %d cases have the expected letter first\n", ok, (int)(sizeof cases / sizeof cases[0]));

    // Timing as on the phone: one call per tap while typing a sentence (cache reused)
    const char *sentence = "I think we should meet at the cafe tomorrow afternoon";
    std::string ctx, word;
    int calls = 0;
    auto t0 = std::chrono::steady_clock::now();
    for(const char *q = sentence; ; q++) {
        if(*q == ' ' || *q == 0) { ctx += (ctx.empty() ? "" : " ") + word; word.clear(); if(!*q) break; continue; }
        NextLetterProbs(state, ctx, word); calls++;
        word += *q;
    }
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    printf("== %d calls, %.1f ms per tap on this CPU (4 threads)\n", calls, ms / calls);
    if(ok < 7) failures++;
    printf("%s\n", failures ? "SOME FAILED" : "ALL PASSED");
    return failures ? 1 : 0;
}
