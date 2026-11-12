package org.futo.inputmethod.latin.uix

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.ByteArrayInputStream
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder

class GgufInfoTest {
    private class Writer {
        val out = ByteArrayOutputStream()
        private fun le(n: Int) = ByteBuffer.allocate(n).order(ByteOrder.LITTLE_ENDIAN)
        fun u32(v: Int) = apply { out.write(le(4).putInt(v).array()) }
        fun u64(v: Long) = apply { out.write(le(8).putLong(v).array()) }
        fun str(s: String) = apply { val b = s.toByteArray(); u64(b.size.toLong()); out.write(b) }
        fun bytes() = out.toByteArray()
    }

    /** Builds a GGUF v3 header with some keys before general.architecture. */
    private fun gguf(arch: String, withPrefixKeys: Boolean = true): ByteArray {
        val w = Writer()
        w.out.write("GGUF".toByteArray())
        w.u32(3)          // version
        w.u64(0)          // tensor count
        w.u64(if (withPrefixKeys) 4 else 1)
        if (withPrefixKeys) {
            w.str("some.u32"); w.u32(4); w.u32(42)
            w.str("some.string"); w.u32(8); w.str("hello")
            w.str("some.array"); w.u32(9); w.u32(8); w.u64(2); w.str("a"); w.str("bb")
        }
        w.str("general.architecture"); w.u32(8); w.str(arch)
        return w.bytes()
    }

    @Test
    fun readsArchitectureAfterOtherKeys() {
        assertEquals("whisper", GgufInfo.readArchitecture(ByteArrayInputStream(gguf("whisper"))))
        assertEquals("llama", GgufInfo.readArchitecture(ByteArrayInputStream(gguf("llama", false))))
    }

    @Test
    fun worksWhenMagicAlreadyConsumed() {
        val stream = ByteArrayInputStream(gguf("parakeet"))
        stream.skip(4)
        assertEquals("parakeet", GgufInfo.readArchitecture(stream, alreadyReadMagic = true))
    }

    @Test
    fun classifiesTextVsVoiceModels() {
        assertTrue(GgufInfo.isTextModel("llama"))
        assertFalse(GgufInfo.isTextModel("whisper"))
        assertFalse(GgufInfo.isTextModel("parakeet"))
        // Unknown/unreadable: keep FUTO's old behaviour (typing model)
        assertTrue(GgufInfo.isTextModel(null))
    }

    @Test
    fun notGgufReturnsNull() {
        assertNull(GgufInfo.readArchitecture(ByteArrayInputStream("lmgg1234".toByteArray())))
        assertNull(GgufInfo.readArchitecture(ByteArrayInputStream(ByteArray(2))))
    }
}
