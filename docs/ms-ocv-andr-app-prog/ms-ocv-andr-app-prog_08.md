# 第八章. 故障排除和最佳实践

错误是开发周期中不可避免的一部分——无论是网站还是移动应用程序。有时它们是逻辑错误、语法错误，甚至是粗心大意的错误。花费大量时间进行调试或纠正错误可能会分散你的注意力，并显著影响你的生产力。在本章中，我们将讨论开发者在构建应用程序时面临的一些常见错误。这可以显著减少调试代码所花费的时间。此外，构建高效的应用程序非常重要。本章的后半部分将讨论一些可以提高应用程序性能的指导方针。

# 故障排除错误

本节讨论开发者在构建 Android 应用程序时可能遇到的不同可能的错误，例如权限错误，以及如何使用 **Logcat** 调试代码。

## 权限错误

在 Android 生态系统中，每个应用程序在执行任何涉及用户数据的临界操作（例如使用互联网或摄像头等）之前都需要用户的权限，仅举几个例子。为了确保这一点，应用程序开发者（在这种情况下，即我们）必须请求用户权限以执行任何临界操作。开发者通过在 Android 项目中声明所有必需的权限来在构建应用程序时完成此操作（更多细节将在以下页面中解释）。当从 Play 商店或其他方式安装应用程序时，用户会被提示授予或拒绝应用程序所需的权限。

只有当用户授予所有权限后，应用程序才能在移动设备上安装。这样，用户就会了解应用程序将要使用所有任务、服务和功能，例如使用互联网或在您的手机内存中存储数据。

Android 是如何确保所有必要的权限都已授予的呢？开发者很可能在构建应用程序时忘记声明一些权限。为了处理这种情况，Android 提供了一系列预定义的任务，这些任务在执行之前需要用户权限。在生成应用程序的 APK 时，代码会检查所有这些任务以及相应的权限是否由开发者声明。一旦代码通过这一测试，就会生成一个可用的 APK，可以用于在任意 Android 手机上安装应用程序。甚至在生成 APK 之前，即在构建应用程序的过程中，如果未声明对应任务的权限，调试器会抛出系统异常，并强制应用程序关闭。

所以，关于权限的工作原理就讲到这里，那么你应该如何以及在哪里声明这些权限，以及构建与计算机视觉相关或其它相关应用程序时需要的一些常见权限有哪些呢？

### 小贴士

如果你已经知道如何声明权限，你可以跳过这部分内容，直接进入下一节，即常用权限部分。

在 Android 项目的 `AndroidManifest.xml` 文件中使用 `<uses-permission>` 标签声明应用程序的权限。例如，如果应用程序需要连接到互联网，则相应的权限应写作如下：

```py
<uses-permission android:name="android.permission.INTERNET"/>
```

最终的 `AndroidManifest.xml` 文件应该看起来像这样：

```py
<manifest 
    package="com.example.Application">

    <application android:allowBackup="true" android:label="@string/app_name"
        android:icon="@mipmap/ic_launcher" android:theme="@style/AppTheme">

        <activity
            android:name="com.example.Application.MainActivity"
            android:label="@string/app_name" >
            <intent-filter>
                <action android:name="android.intent.action.MAIN" />
                <category android:name="android.intent.category.LAUNCHER" />
            </intent-filter>
        </activity>

    </application>
    <uses-permission android:name="android.permission.INTERNET"/>
</manifest>
```

### 注意

注意：权限是在 `<application>` 标签内添加的，而不是在 `<activity>` 标签内。

声明此权限后，你的应用程序将能够使用手机的互联网连接。

### 注意

有关系统和用户权限的更多信息，请参阅 [`developer.android.com/guide/topics/security/permissions.html`](http://developer.android.com/guide/topics/security/permissions.html)。

现在我们来讨论一些 Android 应用程序可能需要的常见权限。

### 一些常用权限

以下是一些在构建应用程序时常用的权限：

+   **使用互联网的权限**：当应用程序想要访问互联网或创建任何网络套接字时，需要此权限：

    ```py
    <uses-permission android:name="android.permission.INTERNET"/>
    ```

+   **读写外部存储**：当应用程序想要从手机的内部存储或 SD 卡中读取数据时，需要这些权限：

    ```py
    <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"/>
    <uses-permission android:name="android.permission.WRITE_EXTERNAL_STORAGE"/>
    ```

+   **访问设备摄像头**：当应用程序想要访问设备摄像头来拍照或录制视频时，需要此权限：

    ```py
    <uses-permission android:name="android.permission.CAMERA"/>
    ```

+   **设置屏幕方向**：当应用程序想要将屏幕方向从横屏切换到竖屏，或反之时，需要此权限：

    ```py
    <uses-permission android:name="android.permission.SET_ORIENTATION"/>
    ```

+   **读取日志**：这允许应用程序读取低级系统日志文件。在调试应用程序时，这非常有帮助：

    ```py
    <uses-permission android:name="android.permission.READ_LOGS"/>
    ```

这些是一些常见的所需权限。根据应用程序的需求，还需要一些其他权限，例如使用 NFC、蓝牙、清除缓存文件等。

## 使用 Logcat 调试代码

如前所述，调试代码是开发周期的重要组成部分，拥有一个使调试更容易的工具是再好不过了。Logcat 就是这样一个工具，它可以帮助你在代码中添加类似打印的语句来检查变量值或某些函数的输出。由于 Android 应用程序是在手机上而不是在电脑上运行，因此调试 Android 应用程序比较困难。

Android 中的 `Log` 类可以帮助你将消息打印到 Logcat。它还提供了不同的日志记录方法，如 `verbose`、`warn`、`debug`、`error` 和 `information`。以下是将日志记录到 Logcat 的方法定义：

```py
v(String, String) (verbose)
d(String, String) (debug)
i(String, String) (information)
w(String, String) (warning)
e(String, String) (error)
```

以下代码展示了如何使用 `Log` 类的示例。此代码摘自 [`developer.android.com/tools/debugging/debugging-studio.html`](https://developer.android.com/tools/debugging/debugging-studio.html)：

```py
import android.util.Log;
public class MyActivity extends Activity {
    private static final String TAG = MyActivity.class.getSimpleName();
    ...
    @Override
    public void onCreate(Bundle savedInstanceState) {
        if (savedInstanceState != null) {
            Log.d(TAG, "onCreate() Restoring previous state");
            /* restore state */
        } else {
            Log.d(TAG, "onCreate() No saved state available");
            /* initialize app */
        }
    }
}
```

### 注意

更多关于 Logcat 和`Log`类的信息，请参阅[`developer.android.com/tools/debugging/debugging-log.html`](https://developer.android.com/tools/debugging/debugging-log.html)。

# 最佳实践

移动平台不如个人计算机强大，因此开发者在为移动设备构建应用程序时需要格外小心。糟糕的代码可以使你的应用程序变得缓慢，因此，在编写代码时，必须考虑到移动设备的资源限制，例如有限的 RAM、有限的处理能力和小的缓存大小。

这里有一些可能影响应用程序性能的事项，在构建应用程序时应注意：

+   **内存泄漏**：在代码中正确管理变量非常重要。因为大多数代码是用 Java 编写的，开发者不需要在处理内存上花费太多时间，因为 Java 会明确地处理。当使用 C/C++时，处理代码中的变量变得极其重要。

+   **重复数据**：在处理使用数据集来训练机器学习算法的应用程序中的大量数据时，我们应该避免在不同形式中有多个相同数据的副本。例如，如果我们有一个以 Mat 对象形式存在的图像，并且将这个对象复制到一个二维整数数组中，那么我们应该确保删除 Mat 对象，因为它不再需要并且无用地占用空间。这样做不仅有助于你的应用程序，还有助于在后台运行的其他应用程序。更多的空闲缓存空间——可以运行更多后台进程的数量。

+   **网络使用**：这同样是一个非常重要的点。许多应用程序需要通过互联网从中央服务器或甚至与其他手机交换数据。为了两个原因，最小化这些设备之间交换的数据量变得非常重要：首先，需要传输的数据量越少，传输时间就越快。这将使应用程序响应更快，数据使用量也会更少（有时数据使用量可能非常昂贵）。其次，这将减少你的移动设备消耗的电量。

+   **有限的计算能力**：避免不必要的冗余计算。例如，如果你的应用程序在多次迭代中对数组进行一些计算，并且一些计算在不同迭代中重复，尝试将这些计算合并并存储在临时变量中，以便在多次迭代中使用（无需再次计算结果）。这里需要注意的一个重要问题是计算能力和内存容量之间的权衡。可能无法存储在应用程序中可能再次重用的每个计算。这很大程度上取决于应用程序的设计。

上述列表并不全面。在构建应用程序时，还有许多其他重要的事情需要考虑，例如处理图像（对于多媒体应用程序）、在活动之间传输数据以及在你的移动设备和服务器（云基础设施）之间分配工作，这些问题将在以下页面中详细讨论。

## 处理 Android 中的图像

你是否曾经想过 Android 应用程序是如何能够加载如此多的图像同时还能保持流畅运行的呢？在本节中，我们将探讨如何将图像加载到我们的应用程序中并处理它们，而不会影响应用程序的性能。

### 加载图像

在许多应用程序中，我们需要从手机的内存中加载图像；例如，在照片编辑器或包含大量缩略图的活动等应用程序中。这样做的问题是需要加载这些图像到应用程序中的内存量。很多时候，即使是`ImageView`控件也因为内存限制而无法加载图像。因此，为了避免此类问题，在加载之前减小图片的大小总是更好的，Android API 为你提供了实现这一点的简单方法。

以下是在将图像加载到应用程序之前用于压缩或减小图像大小的代码：

```py
public static int calculateInSampleSize(
            BitmapFactory.Options options, int reqWidth, int reqHeight) {
    // Raw height and width of image
    final int height = options.outHeight;
    final int width = options.outWidth;
    int inSampleSize = 1;

    if (height > reqHeight || width > reqWidth) {

        final int halfHeight = height / 2;
        final int halfWidth = width / 2;

        // Calculate the largest inSampleSize value that is a power of 2 and keeps both
        // height and width larger than the requested height and width.
        while ((halfHeight / inSampleSize) > reqHeight
                && (halfWidth / inSampleSize) > reqWidth) {
            inSampleSize *= 2;
        }
    }

    return inSampleSize;
}
```

### 处理图像

市面上有许多多媒体应用程序，为用户提供从更改图像亮度、裁剪、调整大小等多种选项。对于此类应用程序来说，高效处理图像非常重要，这意味着这不应该影响用户体验，应用程序也不应该运行缓慢。为了避免这些问题，Android 允许开发者创建除主 UI 线程之外的其他线程，这些线程可以用于在后台执行计算密集型任务。这样做不会影响应用程序的 UI 线程，也不会使应用程序看起来运行缓慢。

在非 UI 线程上卸载计算的一个简单方法是使用`ASyncTasks`。以下是一个说明如何使用`ASyncTasks`的示例。（此代码摘自[`developer.android.com/training/displaying-bitmaps/process-bitmap.html`](http://developer.android.com/training/displaying-bitmaps/process-bitmap.html)）：

```py
class BitmapWorkerTask extends AsyncTask<Integer, Void, Bitmap> {
    private final WeakReference<ImageView> imageViewReference;
    private int data = 0;

    public BitmapWorkerTask(ImageView imageView) {
        // Use a WeakReference to ensure the ImageView can be garbage collected
        imageViewReference = new WeakReference<ImageView>(imageView);
    }

    // Decode image in background.
    @Override
    protected Bitmap doInBackground(Integer... params) {
        data = params[0];
        return decodeSampledBitmapFromResource(getResources(), data, 100, 100));
    }

    // Once complete, see if ImageView is still around and set bitmap.
    @Override
    protected void onPostExecute(Bitmap bitmap) {
        if (imageViewReference != null && bitmap != null) {
            final ImageView imageView = imageViewReference.get();
            if (imageView != null) {
                imageView.setImageBitmap(bitmap);
            }
        }
    }
}
```

## 处理多个活动之间的数据

在本节中，我们将探讨在多个活动之间以高效方式共享数据的不同方法。实现这一目标有多种方式，每种方法都有其优缺点。

这里有一些在活动之间交换数据的方法：

+   通过 Intent 传输数据

+   使用静态字段

+   使用数据库或文件

### 通过 Intent 传输数据

这是 Android 中在活动之间交换信息最常见的方法之一。

在 Android 中，使用 `Intent` 类启动新的活动。`Intent` 类允许你将数据作为键值对作为额外的数据发送给正在启动的活动。以下是一个演示此功能的示例：

```py
public void launchNewActivity () {
  Intent intent = new Intent(this, NewActivity.class);
  intent.putExtra("Message", "Sending a string to New Activity");
  startActivity(intent);
}
```

在前面的代码中，`NewActivity` 是正在启动的新活动的名称。`putExtra` 函数分别将键和值作为第一个和第二个参数。

下一步是在启动的活动中检索数据。执行此操作的代码如下：

```py
Intent intent = getIntent();
String message = intent.getStringExtra("Message");
```

`getStringExtra` 函数获取函数中作为参数传递的键对应的值；在这种情况下，`Message`。

### 使用静态字段

在 Android 中，另一种在活动之间交换数据的方法是使用静态字段。使用静态字段的主要思想是它们在整个程序的生命周期中是持久的，并且不需要任何对象来引用它们。

这里是一个使用静态字段进行数据交换的类的示例：

```py
public class StorageClass {
  private static String data;
  public static String getData() {return data;}
  public static String setData(String data) {this.data = data;}
}
```

`StorageClass` 函数有一个静态字段 `data`，它将存储需要传递到新活动中的信息。

从启动活动：

```py
StorageClass.setData("Here is a message");
```

在启动的活动：

```py
String data = StorageClass.getData();
```

### 使用数据库或文件

这是交换活动之间数据最复杂的方式之一。其背后的想法是使用 SQLite 或任何其他数据库框架设置数据库，并将其用作活动之间的共享资源。这种方法需要你编写更多的代码。此外，从数据库写入和读取可能比其他提到的技术要慢。然而，当涉及到共享大量数据而不是简单的字符串或整数时，这种方法更好。这些是一些可以用于在多个活动之间高效交换数据的技巧。

# 摘要

本章总结了开发者在构建基于 Android 平台的计算机视觉应用程序时可能遇到的所有可能的权限和错误。我们还探讨了可以使应用程序性能更好的最佳实践。在下一章中，我们将尝试巩固迄今为止所学的一切，从头开始构建一个简单而强大的应用程序。
