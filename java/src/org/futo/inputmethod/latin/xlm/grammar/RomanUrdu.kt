package org.futo.inputmethod.latin.xlm.grammar

/**
 * Detects Roman Urdu (Urdu written with English letters).
 *
 * Used to:
 *  - skip grammar checking on sentences that are mostly Roman Urdu
 *  - keep Roman Urdu words in their own learned vocabulary, so autocorrect
 *    doesn't "fix" them into English and they don't pollute English predictions
 *
 * This is a lightweight word-list heuristic. Words that are also common English
 * words ("to", "the", "me", "us", "par", "say", ...) are deliberately excluded,
 * because counting them would misfire on normal English sentences.
 */
object RomanUrdu {
    /** Common Roman Urdu words, including frequent spelling variants. */
    val COMMON_WORDS: Set<String> = setOf(
        // Verbs / auxiliaries
        "hai", "hain", "hy", "hay", "hei", "tha", "thi", "thay", "thin", "hoga", "hogi", "honge",
        "ho", "hoon", "hun", "hua", "hui", "huay", "hue", "raha", "rahi", "rahe", "rahay",
        "gaya", "gayi", "gai", "gaye", "gye", "gya", "karo", "karna", "karta", "karti", "karte",
        "kar", "kiya", "kia", "kya", "kyaa", "krna", "krta", "krti", "krte", "kro", "kr",
        "jao", "jana", "jata", "jati", "jate", "aao", "ao", "ana", "aata", "aati", "aate", "aya", "ayi",
        "dekho", "dekh", "dekha", "dekhna", "suno", "suna", "bolo", "bola", "boli", "bolna",
        "chalo", "chal", "chala", "chali", "lena", "lelo", "dena", "dedo", "do", "diya", "dia",
        "sakta", "sakti", "sakte", "chahiye", "chahye", "chahta", "chahti", "pata", "lagta", "lagti",
        "jaunga", "jaungi", "jayega", "jayegi", "karunga", "karungi", "karega", "karegi",
        "milte", "milta", "milti", "mila", "mili", "rakho", "rakh", "baitho", "betho", "uthao",
        // Pronouns
        "main", "mein", "mai", "mujhe", "mujhay", "mjhe", "mera", "meri", "mere", "humein", "hamein",
        "hum", "hamara", "hamari", "hamare", "tum", "tumhe", "tumhein", "tumhara", "tumhari", "tumhare",
        "aap", "ap", "apka", "apki", "apke", "aapka", "aapki", "aapke", "apna", "apni", "apne",
        "yeh", "ye", "woh", "wo", "wu", "unka", "unki", "unke", "uska", "uski", "uske", "isko", "usko",
        "inko", "unko", "inhe", "unhe", "unhein", "kisi", "koi", "kuch", "kuchh", "sab", "sabko",
        // Postpositions / particles
        "ka", "ki", "ke", "ko", "se", "sy", "pe", "mai", "tak", "liye", "lye", "liay", "saath", "sath",
        "wala", "wali", "wale", "waly", "bhi", "bhee", "hi", "toh", "tou", "na", "nahi", "nahin", "nai",
        "nhi", "nahe", "mat", "haan", "han", "jee", "ji", "acha", "achha", "accha", "theek", "thik", "thek",
        // Question words / conjunctions / adverbs
        "kyun", "kyon", "kyu", "kab", "kahan", "kidhar", "kaise", "kaisay", "kesa", "kaisa", "kaisi",
        "kitna", "kitni", "kitne", "kaun", "kon", "jab", "tab", "phir", "fir", "aur", "or", "lekin",
        "magar", "agar", "warna", "kyunke", "kyunki", "kiyunke", "abhi", "abi", "ab", "kal", "aaj",
        "parson", "bohat", "bohot", "bahut", "boht", "zyada", "ziyada", "kam", "thora", "thoda",
        "thori", "thodi", "bilkul", "zaroor", "shayad", "sirf", "bas", "yahan", "wahan", "idhar", "udhar",
        // Common nouns / adjectives / interjections
        "yaar", "yar", "bhai", "behen", "ammi", "abbu", "khana", "pani", "ghar", "kaam", "kam", "baat",
        "baatein", "dost", "log", "logon", "din", "raat", "subah", "shaam", "waqt", "pehle", "baad",
        "naya", "nayi", "purana", "achi", "achay", "ache", "bura", "buri", "khush", "pyar", "dil",
        "shukriya", "mehrbani", "inshallah", "mashallah", "alhamdulillah", "salam", "assalam",
    )

    /**
     * Words that look like Roman Urdu but are also common English words. These are never
     * counted toward Roman Urdu detection so that English text isn't misclassified.
     */
    val ENGLISH_COLLISIONS: Set<String> = setOf(
        "to", "the", "me", "us", "par", "say", "so", "do", "or", "hi", "main", "ban", "car", "tab",
        "bus", "log", "din", "ab", "pe", "han", "mat", "fir", "sab", "ana", "kit", "jab", "bas",
        "kal", "kam", "dil", "ye", "pata", "chal", "kon",
    )

    private val COUNTED_WORDS: Set<String> = COMMON_WORDS - ENGLISH_COLLISIONS

    /** Strong markers: almost never appear in English text. */
    private val STRONG_MARKERS: Set<String> = setOf(
        "hai", "hain", "nahi", "nahin", "nhi", "kya", "kyun", "kyon", "mujhe", "tum", "aap", "yeh",
        "woh", "bhi", "acha", "achha", "accha", "theek", "thik", "raha", "rahi", "gaya", "karna",
        "karo", "chahiye", "bohat", "bahut", "kaise", "yaar", "tha", "thi", "hoga", "mera", "meri",
        "tumhara", "aur", "lekin", "abhi", "kuch", "koi", "wala", "wali", "liye", "sath", "saath",
        "se", "ko", "ka", "ke", "ki", "toh", "mein", "hum", "ho",
    ) - ENGLISH_COLLISIONS

    private val WORD_REGEX = Regex("[A-Za-z']+")

    fun words(text: String): List<String> =
        WORD_REGEX.findAll(text).map { it.value.lowercase().trim('\'') }.filter { it.isNotEmpty() }.toList()

    /** True if this single word is a known Roman Urdu word (regardless of collisions). */
    fun isRomanUrduWord(word: String): Boolean = word.lowercase().trim('\'') in COMMON_WORDS

    /**
     * Fraction (0..1) of words in [text] that look like Roman Urdu.
     */
    fun romanUrduRatio(text: String): Float {
        val ws = words(text)
        if (ws.isEmpty()) return 0f
        val hits = ws.count { it in COUNTED_WORDS }
        return hits.toFloat() / ws.size
    }

    /**
     * True if the sentence is mostly Roman Urdu. The user's mixed sentences are
     * mostly-Urdu with a few English words, so a fairly low threshold is used, but at
     * least two Roman Urdu words must be present so a single coincidence can't trigger it.
     */
    fun isMostlyRomanUrdu(text: String): Boolean {
        val ws = words(text)
        if (ws.isEmpty()) return false
        val hits = ws.count { it in COUNTED_WORDS }
        val strongHits = ws.count { it in STRONG_MARKERS }
        if (hits < 2 && strongHits < 2) return false
        val ratio = hits.toFloat() / ws.size
        return ratio >= 0.35f || (strongHits >= 2 && ratio >= 0.25f)
    }
}
