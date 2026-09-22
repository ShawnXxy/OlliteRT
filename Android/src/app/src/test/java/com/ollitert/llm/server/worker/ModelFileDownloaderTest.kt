package com.ollitert.llm.server.worker

import com.ollitert.llm.server.data.download.ModelScopeFallback
import com.ollitert.llm.server.data.storage.modelScopeStagingFile
import java.io.ByteArrayInputStream
import java.io.File
import java.net.HttpURLConnection
import java.net.SocketTimeoutException
import java.net.URL
import java.security.MessageDigest
import javax.net.ssl.SSLHandshakeException
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.test.runTest
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class ModelFileDownloaderTest {
  @get:Rule val temp = TemporaryFolder()
  private val primary = "https://huggingface.co/org/model/resolve/revision/model.litertlm"
  private val mirrorUrl = "https://modelscope.cn/models/org/model/resolve/pinned/model.litertlm"
  private val bytes = "LITERTLMtest model bytes".toByteArray()
  private val hash = MessageDigest.getInstance("SHA-256").digest(bytes)
    .joinToString("") { "%02x".format(it) }
  private val mirror = ModelScopeFallback(mirrorUrl, bytes.size.toLong(), hash)

  private open class Response(
    url: URL,
    private val body: ByteArray,
    private val status: Int = 200,
    private val headers: Map<String, String> = emptyMap(),
  ) : HttpURLConnection(url) {
    override fun connect() = Unit
    override fun disconnect() = Unit
    override fun usingProxy() = false
    override fun getResponseCode() = status
    override fun getHeaderField(name: String): String? = headers[name]
    override fun getInputStream() = ByteArrayInputStream(body)
  }

  private fun staging(): File = File(temp.root, "model.olliterttmp")

  @Test
  fun successfulPrimaryNeverContactsMirror() = runTest {
    val requests = mutableListOf<String>()
    val downloader = ModelFileDownloader { url ->
      requests.add(url.toString())
      Response(url, bytes)
    }
    val result = downloader.download(primary, staging(), mirror, { true })
    assertArrayEquals(bytes, result.readBytes())
    assertEquals(listOf(primary), requests)
  }

  @Test
  fun networkFailureRequiresConsentWithoutContactingMirror() = runTest {
    val requests = mutableListOf<String>()
    val downloader = ModelFileDownloader { url ->
      requests.add(url.toString())
      throw SocketTimeoutException("timed out")
    }
    try {
      downloader.download(primary, staging(), mirror, { false })
      fail("Expected consent request")
    } catch (failure: ModelScopeConsentRequiredException) {
      assertTrue(failure.primaryError.isNotBlank())
    }
    assertEquals(listOf(primary), requests)
  }

  @Test
  fun alternateStartsAtZeroAndNeverReceivesPrimaryCredentials() = runTest {
    val staging = staging().apply { writeBytes("old primary bytes".toByteArray()) }
    lateinit var mirrorRequest: Response
    val downloader = ModelFileDownloader { url ->
      if (url.toString() == primary) throw SocketTimeoutException("timed out")
      Response(url, bytes).also { mirrorRequest = it }
    }
    val result = downloader.download(primary, staging, mirror, { true }, accessToken = "hf-private")
    assertArrayEquals(bytes, result.readBytes())
    assertNull(mirrorRequest.getRequestProperty("Authorization"))
    assertNull(mirrorRequest.getRequestProperty("Range"))
    assertFalse(staging.exists())
  }

  @Test
  fun mirrorRedirectDoesNotReceivePrimaryCredentials() = runTest {
    lateinit var cdnRequest: Response
    val downloader = ModelFileDownloader { url ->
      when (url.host) {
        "huggingface.co" -> throw SocketTimeoutException()
        "modelscope.cn" -> Response(
          url, byteArrayOf(), 302, mapOf("Location" to "https://cdn-lfs-cn-1.modelscope.cn/file"),
        )
        else -> Response(url, bytes).also { cdnRequest = it }
      }
    }
    downloader.download(primary, staging(), mirror, { true }, accessToken = "hf-private")
    assertNull(cdnRequest.getRequestProperty("Authorization"))
  }

  @Test
  fun savedMirrorPartialResumesOnlyAgainstTheSameMirror() = runTest {
    val staging = staging()
    modelScopeStagingFile(staging, mirror.sha256).writeBytes(bytes.take(8).toByteArray())
    lateinit var mirrorRequest: Response
    val downloader = ModelFileDownloader { url ->
      if (url.toString() == primary) throw SocketTimeoutException()
      Response(
        url, bytes.drop(8).toByteArray(), 206,
        mapOf("Content-Range" to "bytes 8-${bytes.lastIndex}/${bytes.size}"),
      ).also { mirrorRequest = it }
    }
    val result = downloader.download(primary, staging, mirror, { true })
    assertEquals("bytes=8-", mirrorRequest.getRequestProperty("Range"))
    assertArrayEquals(bytes, result.readBytes())
  }

  @Test
  fun ignoredRangeRestartsInsteadOfAppending() = runTest {
    val staging = staging().apply { writeBytes(bytes.take(8).toByteArray()) }
    val downloader = ModelFileDownloader { url -> Response(url, bytes) }
    val result = downloader.download(primary, staging, mirror, { true })
    assertArrayEquals(bytes, result.readBytes())
  }

  @Test
  fun bothProviderFailuresAreReportedAndAttemptsAreBounded() = runTest {
    val requests = mutableListOf<String>()
    val downloader = ModelFileDownloader { url ->
      requests.add(url.toString())
      Response(url, byteArrayOf(), 503)
    }
    try {
      downloader.download(primary, staging(), mirror, { true })
      fail("Expected both failures")
    } catch (failure: ModelScopeDownloadException) {
      assertTrue(failure.message.orEmpty().contains("Hugging Face"))
      assertTrue(failure.message.orEmpty().contains("ModelScope"))
    }
    assertEquals(listOf(primary, mirrorUrl), requests)
  }

  @Test
  fun authenticationNotFoundAndCertificateErrorsNeverFallBack() = runTest {
    for (code in listOf(401, 403, 404)) {
      val requests = mutableListOf<String>()
      val downloader = ModelFileDownloader { url ->
        requests.add(url.toString())
        Response(url, byteArrayOf(), code)
      }
      try {
        downloader.download(primary, staging(), mirror, { true })
        fail("Expected HTTP failure")
      } catch (_: com.ollitert.llm.server.data.download.DownloadHttpException) { }
      assertEquals(listOf(primary), requests)
    }
    val requests = mutableListOf<String>()
    val downloader = ModelFileDownloader { url ->
      requests.add(url.toString())
      throw SSLHandshakeException("certificate rejected")
    }
    try {
      downloader.download(primary, staging(), mirror, { true })
      fail("Expected TLS failure")
    } catch (_: SSLHandshakeException) { }
    assertEquals(listOf(primary), requests)
  }

  @Test
  fun changedMirrorBytesCannotBePublishedDespiteMatchingSizeAndHeader() = runTest {
    val corrupt = bytes.copyOf().apply { this[lastIndex] = 0 }
    val downloader = ModelFileDownloader { url ->
      if (url.toString() == primary) throw SocketTimeoutException()
      Response(url, corrupt)
    }
    try {
      downloader.download(primary, staging(), mirror, { true })
      fail("Expected checksum failure")
    } catch (failure: ModelScopeDownloadException) {
      assertTrue(failure.cause is DownloadIntegrityException)
    }
    assertFalse(modelScopeStagingFile(staging(), mirror.sha256).exists())
  }

  @Test
  fun revokedConsentPreventsSkippingToTheMirror() = runTest {
    val requests = mutableListOf<String>()
    val downloader = ModelFileDownloader { url ->
      requests.add(url.toString())
      Response(url, bytes)
    }
    downloader.download(primary, staging(), mirror, { false }, primaryError = "earlier timeout")
    assertEquals(listOf(primary), requests)
  }

  @Test
  fun approvedPreflightFailureDoesNotRepeatThePrimaryAttempt() = runTest {
    val requests = mutableListOf<String>()
    val downloader = ModelFileDownloader { url ->
      requests.add(url.toString())
      Response(url, bytes)
    }
    downloader.download(primary, staging(), mirror, { true }, primaryError = "DNS failed")
    assertEquals(listOf(mirrorUrl), requests)
  }

  @Test
  fun cancellationNeverContactsTheAlternate() = runTest {
    val requests = mutableListOf<String>()
    val downloader = ModelFileDownloader { url ->
      requests.add(url.toString())
      throw CancellationException("cancelled")
    }
    try {
      downloader.download(primary, staging(), mirror, { true })
      fail("Expected cancellation")
    } catch (_: CancellationException) { }
    assertEquals(listOf(primary), requests)
  }

  @Test
  fun failedLocalWriteNeverContactsTheAlternate() = runTest {
    val requests = mutableListOf<String>()
    val downloader = ModelFileDownloader { url ->
      requests.add(url.toString())
      Response(url, bytes)
    }
    val missingParent = File(temp.root, "missing/model.olliterttmp")
    try {
      downloader.download(primary, missingParent, mirror, { true })
      fail("Expected local file failure")
    } catch (_: java.io.FileNotFoundException) { }
    assertEquals(listOf(primary), requests)
  }

  @Test
  fun truncatedPrimaryFallsBackWithoutCombiningBytes() = runTest {
    val downloader = ModelFileDownloader { url ->
      Response(url, if (url.toString() == primary) bytes.take(10).toByteArray() else bytes)
    }
    val result = downloader.download(primary, staging(), mirror, { true })
    assertArrayEquals(bytes, result.readBytes())
    assertEquals(modelScopeStagingFile(staging(), mirror.sha256), result)
  }

  @Test
  fun invalidMirrorResumeRangeIsRejectedAndItsStagingFileRemoved() = runTest {
    val staging = staging()
    modelScopeStagingFile(staging, mirror.sha256).writeBytes(bytes.take(8).toByteArray())
    val downloader = ModelFileDownloader { url ->
      if (url.toString() == primary) throw SocketTimeoutException()
      Response(url, bytes, 206, mapOf("Content-Range" to "bytes 0-${bytes.lastIndex}/${bytes.size}"))
    }
    try {
      downloader.download(primary, staging, mirror, { true })
      fail("Expected resume mismatch")
    } catch (failure: ModelScopeDownloadException) {
      assertTrue(failure.cause is DownloadIntegrityException)
    }
    assertFalse(modelScopeStagingFile(staging, mirror.sha256).exists())
  }

  @Test
  fun completedMirrorStagingIsVerifiedBeforeReuse() = runTest {
    val staging = staging()
    modelScopeStagingFile(staging, mirror.sha256).writeBytes(bytes)
    val requests = mutableListOf<String>()
    val downloader = ModelFileDownloader { url ->
      requests.add(url.toString())
      throw SocketTimeoutException()
    }
    val result = downloader.download(primary, staging, mirror, { true })
    assertArrayEquals(bytes, result.readBytes())
    assertEquals(listOf(primary), requests)
  }

  @Test
  fun downgradeRedirectIsRejectedWithoutFallback() = runTest {
    val requests = mutableListOf<String>()
    val downloader = ModelFileDownloader { url ->
      requests.add(url.toString())
      Response(url, byteArrayOf(), 302, mapOf("Location" to "http://example.test/model"))
    }
    try {
      downloader.download(primary, staging(), mirror, { true })
      fail("Expected unsafe redirect rejection")
    } catch (_: java.net.ProtocolException) { }
    assertEquals(listOf(primary), requests)
  }

  @Test
  fun changingTheMirrorArtifactNeverReusesItsOldPartialFile() = runTest {
    val staging = staging()
    val oldMirror = mirror.copy(url = "$mirrorUrl-old", sha256 = "0".repeat(64))
    val first = ModelFileDownloader { url ->
      if (url.toString() == primary) throw SocketTimeoutException()
      Response(url, bytes.take(10).toByteArray())
    }
    try {
      first.download(primary, staging, oldMirror, { true })
      fail("Expected incomplete old mirror")
    } catch (_: ModelScopeDownloadException) { }

    lateinit var mirrorRequest: Response
    val second = ModelFileDownloader { url ->
      if (url.toString() == primary) throw SocketTimeoutException()
      Response(url, bytes).also { mirrorRequest = it }
    }
    val result = second.download(primary, staging, mirror, { true })
    assertArrayEquals(bytes, result.readBytes())
    assertNull(mirrorRequest.getRequestProperty("Range"))
  }
}
