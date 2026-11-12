package org.futo.voiceinput.shared.types

import android.content.Context
import androidx.annotation.StringRes
import org.futo.voiceinput.shared.ggml.WhisperGGML
import java.io.File
import java.io.FileOutputStream
import java.io.IOException


/**
 * The voice engine (transcribe.cpp) loads models from a file path, so the built-in model
 * (stored inside the APK's assets) is copied out once to the code cache directory.
 */
@Throws(IOException::class)
private fun extractAsset(context: Context, assetName: String): File {
    val file = File(context.codeCacheDir, "_FUTOASSET_$assetName")
    val expectedSize = try {
        context.assets.openFd(assetName).use { it.length }
    } catch (e: IOException) {
        -1L
    }
    if (file.isFile && (expectedSize < 0 || file.length() == expectedSize)) return file

    val tmp = File(context.codeCacheDir, "_FUTOASSET_$assetName.tmp")
    context.assets.open(assetName).use { input ->
        FileOutputStream(tmp).use { output -> input.copyTo(output) }
    }
    tmp.renameTo(file)
    return file
}

// Maybe add `val languages: Set<Language>`
interface ModelLoader {
    @get:StringRes
    val name: Int

    fun exists(context: Context): Boolean
    fun getRequiredDownloadList(context: Context): List<String>

    fun loadGGML(context: Context): WhisperGGML

    fun key(context: Context): Any
}

internal class ModelBuiltInAsset(
    override val name: Int,
    val ggmlFile: String
) : ModelLoader {
    override fun exists(context: Context): Boolean {
        return true
    }

    override fun getRequiredDownloadList(context: Context): List<String> {
        return listOf()
    }

    override fun loadGGML(context: Context): WhisperGGML {
        return WhisperGGML(extractAsset(context, ggmlFile))
    }

    override fun key(context: Context): Any {
        return "BuiltIn$ggmlFile"
    }
}

internal class ModelDownloadable(
    override val name: Int,
    val ggmlFile: String,
    val checksum: String
) : ModelLoader {
    override fun exists(context: Context): Boolean {
        return getRequiredDownloadList(context).isEmpty()
    }

    override fun getRequiredDownloadList(context: Context): List<String> {
        return listOf(ggmlFile).filter {
            !File(context.filesDir, it).exists()
        }
    }

    override fun loadGGML(context: Context): WhisperGGML {
        return WhisperGGML(File(context.filesDir, ggmlFile))
    }

    override fun key(context: Context): Any {
        return "Downloadable$ggmlFile$checksum"
    }
}

public class ModelFileFile(
    override val name: Int,
    val file: File,
) : ModelLoader {
    override fun exists(context: Context): Boolean {
        return file.exists()
    }

    override fun getRequiredDownloadList(context: Context): List<String> {
        return listOf()
    }

    override fun loadGGML(context: Context): WhisperGGML {
        return WhisperGGML(file)
    }

    override fun key(context: Context): Any {
        return "File${file.absolutePath}"
    }
}
