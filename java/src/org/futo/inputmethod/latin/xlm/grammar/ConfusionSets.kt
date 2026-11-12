package org.futo.inputmethod.latin.xlm.grammar

/**
 * Commonly confused English words. For each word, lists the alternatives the grammar
 * checker should consider. The language model decides which one fits the sentence;
 * these lists only limit what it is allowed to change, which keeps edits small and
 * predictable (it can never rewrite your wording).
 */
object ConfusionSets {
    private val GROUPS: List<List<String>> = listOf(
        listOf("their", "there", "they're"),
        listOf("its", "it's"),
        listOf("your", "you're"),
        listOf("whose", "who's"),
        listOf("then", "than"),
        listOf("to", "too"),
        listOf("lose", "loose"),
        listOf("affect", "effect"),
        listOf("accept", "except"),
        listOf("a", "an"),
        listOf("were", "we're"),
        listOf("weather", "whether"),
        listOf("quiet", "quite"),
        listOf("advice", "advise"),
        listOf("breath", "breathe"),
        listOf("passed", "past"),
        listOf("principal", "principle"),
        listOf("stationary", "stationery"),
        listOf("complement", "compliment"),
        listOf("aloud", "allowed"),
        listOf("brake", "break"),
        listOf("peace", "piece"),
        listOf("led", "lead"),
        listOf("of", "have"), // "could of" -> "could have"

        // Subject-verb agreement ("he go" -> "he goes", "they was" -> "they were")
        listOf("is", "are"),
        listOf("was", "were"),
        listOf("has", "have"),
        listOf("does", "do"),
        listOf("doesn't", "don't"),
        listOf("isn't", "aren't"),
        listOf("wasn't", "weren't"),
        listOf("hasn't", "haven't"),

        // Common past-participle mistakes ("I seen" -> "I saw", "have went" -> "have gone")
        listOf("seen", "saw"),
        listOf("went", "gone"),
        listOf("did", "done"),
        listOf("came", "come"),
        listOf("ran", "run"),
        listOf("began", "begun"),
        listOf("drank", "drunk"),
        listOf("wrote", "written"),
        listOf("ate", "eaten"),
        listOf("took", "taken"),
        listOf("gave", "given"),
        listOf("broke", "broken"),
        listOf("spoke", "spoken"),
        listOf("chose", "chosen"),
        listOf("knew", "known"),
    )

    /** Base verb forms that get a 3rd-person-singular alternative ("go" <-> "goes"). */
    private val AGREEMENT_VERBS: List<String> = listOf(
        "go", "do", "want", "need", "like", "know", "think", "say", "make", "take", "come", "see",
        "get", "give", "look", "use", "find", "tell", "ask", "work", "seem", "feel", "try", "leave",
        "call", "keep", "let", "begin", "help", "talk", "turn", "start", "show", "hear", "play",
        "run", "move", "live", "believe", "bring", "happen", "write", "sit", "stand", "lose", "pay",
        "meet", "include", "continue", "learn", "change", "lead", "understand", "watch", "follow",
        "stop", "create", "speak", "read", "spend", "grow", "open", "walk", "win", "teach", "offer",
        "remember", "love", "consider", "appear", "buy", "wait", "serve", "die", "send", "expect",
        "build", "stay", "fall", "cut", "reach", "kill", "remain", "eat", "drink", "sleep", "drive",
        "hate", "mean", "care", "matter", "cost", "hurt", "miss", "wish", "hope", "fix", "wash",
    )

    private fun thirdPerson(verb: String): String = when {
        verb == "do" -> "does"
        verb == "go" -> "goes"
        verb.endsWith("y") && verb.length > 2 && verb[verb.length - 2] !in "aeiou" -> verb.dropLast(1) + "ies"
        verb.endsWith("s") || verb.endsWith("sh") || verb.endsWith("ch") || verb.endsWith("x") || verb.endsWith("z") || verb.endsWith("o") -> verb + "es"
        else -> verb + "s"
    }

    private val MAP: Map<String, List<String>> = buildMap<String, MutableList<String>> {
        fun link(group: List<String>) {
            for (w in group) {
                val list = getOrPut(w) { mutableListOf() }
                for (other in group) if (other != w && other !in list) list.add(other)
            }
        }
        GROUPS.forEach { link(it) }
        AGREEMENT_VERBS.forEach { link(listOf(it, thirdPerson(it))) }
    }

    /** Lowercase alternatives for [word], or empty if it isn't in any confusion set. */
    fun alternativesFor(word: String): List<String> = MAP[word.lowercase()] ?: emptyList()

    fun contains(word: String): Boolean = MAP.containsKey(word.lowercase())

    /** Applies the capitalisation pattern of [original] to [replacement]. */
    fun matchCase(original: String, replacement: String): String = when {
        original.length > 1 && original.all { !it.isLetter() || it.isUpperCase() } -> replacement.uppercase()
        original.firstOrNull()?.isUpperCase() == true -> replacement.replaceFirstChar { it.uppercaseChar() }
        else -> replacement
    }

    /** Exposed for tests. */
    internal fun thirdPersonForTest(verb: String) = thirdPerson(verb)
}
