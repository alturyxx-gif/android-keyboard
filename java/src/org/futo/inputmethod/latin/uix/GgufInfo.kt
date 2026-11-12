package org.futo.inputmethod.latin.uix

import java.io.DataInputStream
import java.io.InputStream

/**
 * Minimal GGUF header reader (custom build).
 *
 * Both the keyboard's typing model and the new voice models (Whisper large-v3-turbo,
 * Parakeet, Nemotron, ...) are GGUF files, so the file magic alone can't tell them apart.
 * This reads the key/value metadata until it finds `general.architecture`.
 */
object GgufInfo {
    private const val MAGIC = 0x46554747 // "GGUF" little-endian

    // GGUF value types
    private const val T_UINT8 = 0; private const val T_INT8 = 1
    private const val T_UINT16 = 2; private const val T_INT16 = 3
    private const val T_UINT32 = 4; private const val T_INT32 = 5
    private const val T_FLOAT32 = 6; private const val T_BOOL = 7
    private const val T_STRING = 8; private const val T_ARRAY = 9
    private const val T_UINT64 = 10; private const val T_INT64 = 11
    private const val T_FLOAT64 = 12

    /** Architectures that are the keyboard's typing (text) model. */
    val TEXT_MODEL_ARCHITECTURES = setOf("llama")

    private class LE(input: InputStream) {
        private val d = DataInputStream(input)
        fun u8(): Int = d.readUnsignedByte()
        fun u32(): Long = (u8().toLong()) or (u8().toLong() shl 8) or (u8().toLong() shl 16) or (u8().toLong() shl 24)
        fun u64(): Long = u32() or (u32() shl 32)
        fun skip(n: Long) {
            var left = n
            while (left > 0) {
                val s = d.skip(left)
                if (s <= 0) { d.readByte(); left-- } else left -= s
            }
        }
        fun bytes(n: Int): ByteArray = ByteArray(n).also { d.readFully(it) }
    }

    private fun sizeOf(type: Int): Long = when (type) {
        T_UINT8, T_INT8, T_BOOL -> 1
        T_UINT16, T_INT16 -> 2
        T_UINT32, T_INT32, T_FLOAT32 -> 4
        T_UINT64, T_INT64, T_FLOAT64 -> 8
        else -> -1
    }

    private fun skipValue(r: LE, type: Int, v1: Boolean) {
        when (type) {
            T_STRING -> r.skip(if (v1) r.u32() else r.u64())
            T_ARRAY -> {
                val itemType = r.u32().toInt()
                val count = if (v1) r.u32() else r.u64()
                val sz = sizeOf(itemType)
                if (sz > 0) r.skip(sz * count) else repeat(count.toInt()) { skipValue(r, itemType, v1) }
            }
            else -> {
                val sz = sizeOf(type)
                require(sz > 0) { "Unknown GGUF type $type" }
                r.skip(sz)
            }
        }
    }

    /** Returns `general.architecture`, or null if this isn't a readable GGUF file. */
    fun readArchitecture(input: InputStream, alreadyReadMagic: Boolean = false, maxKeys: Int = 256): String? = try {
        val r = LE(input)
        if (!alreadyReadMagic && r.u32().toInt() != MAGIC) null else {
            val version = r.u32().toInt()
            val v1 = version == 1
            if (v1) { r.u32(); } else { r.u64() }              // tensor count
            val kvCount = if (v1) r.u32() else r.u64()
            var result: String? = null
            for (i in 0 until minOf(kvCount, maxKeys.toLong())) {
                val keyLen = if (v1) r.u32() else r.u64()
                if (keyLen > 4096) break
                val key = String(r.bytes(keyLen.toInt()), Charsets.UTF_8)
                val type = r.u32().toInt()
                if (key == "general.architecture" && type == T_STRING) {
                    val len = if (v1) r.u32() else r.u64()
                    if (len > 4096) break
                    result = String(r.bytes(len.toInt()), Charsets.UTF_8)
                    break
                }
                skipValue(r, type, v1)
            }
            result
        }
    } catch (e: Exception) {
        null
    }

    fun isTextModel(architecture: String?): Boolean =
        architecture == null || architecture in TEXT_MODEL_ARCHITECTURES
}
