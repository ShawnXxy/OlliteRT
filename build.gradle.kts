import java.util.Properties
import org.gradle.api.tasks.wrapper.Wrapper

val androidWrapperProperties = Properties().apply {
  file("Android/src/gradle/wrapper/gradle-wrapper.properties").inputStream().use {
    load(it)
  }
}

tasks.named<Wrapper>("wrapper") {
  distributionUrl = requireNotNull(androidWrapperProperties.getProperty("distributionUrl")) {
    "The Android Gradle wrapper must define distributionUrl."
  }
}
