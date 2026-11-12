// Stubs for JNI helpers that the included JNI sources reference but the tests never call.
#include <jni.h>
namespace latinime {
int registerNativeMethods(JNIEnv *, const char *, const JNINativeMethod *, int) { return 0; }
}
