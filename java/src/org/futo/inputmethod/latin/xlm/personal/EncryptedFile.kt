package org.futo.inputmethod.latin.xlm.personal

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Log
import java.io.File
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

/**
 * Small helper that stores text files encrypted with an AES-256-GCM key that lives in the
 * Android Keystore (hardware-backed on the S25). The key never leaves the secure hardware,
 * so the typing history on disk is unreadable without this app on this phone.
 *
 * File format: [12-byte IV][ciphertext+tag]
 */
object EncryptedFile {
    private const val TAG = "EncryptedFile"
    private const val KEY_ALIAS = "futo_keyboard_custom_personalisation_v1"
    private const val ANDROID_KEYSTORE = "AndroidKeyStore"
    private const val TRANSFORMATION = "AES/GCM/NoPadding"
    private const val IV_SIZE = 12
    private const val TAG_BITS = 128

    private fun getOrCreateKey(): SecretKey {
        val ks = KeyStore.getInstance(ANDROID_KEYSTORE).apply { load(null) }
        (ks.getEntry(KEY_ALIAS, null) as? KeyStore.SecretKeyEntry)?.let { return it.secretKey }

        val gen = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, ANDROID_KEYSTORE)
        gen.init(
            KeyGenParameterSpec.Builder(
                KEY_ALIAS,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
            )
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setKeySize(256)
                .build()
        )
        return gen.generateKey()
    }

    fun write(file: File, text: String): Boolean = try {
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, getOrCreateKey())
        val iv = cipher.iv
        val ct = cipher.doFinal(text.toByteArray(Charsets.UTF_8))

        val tmp = File(file.parentFile, file.name + ".tmp")
        tmp.outputStream().use {
            it.write(iv)
            it.write(ct)
        }
        tmp.renameTo(file)
    } catch (e: Exception) {
        Log.e(TAG, "Failed to write ${file.name}", e)
        false
    }

    fun read(file: File): String? {
        if (!file.isFile) return null
        return try {
            val bytes = file.readBytes()
            if (bytes.size <= IV_SIZE) return null
            val cipher = Cipher.getInstance(TRANSFORMATION)
            cipher.init(
                Cipher.DECRYPT_MODE,
                getOrCreateKey(),
                GCMParameterSpec(TAG_BITS, bytes, 0, IV_SIZE)
            )
            String(cipher.doFinal(bytes, IV_SIZE, bytes.size - IV_SIZE), Charsets.UTF_8)
        } catch (e: Exception) {
            Log.e(TAG, "Failed to read ${file.name}", e)
            null
        }
    }
}
