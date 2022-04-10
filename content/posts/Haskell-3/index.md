---
weight: 4
title: "Haskell 学习笔记四"
date: 2022-03-04T21:57:40+08:00
lastmod: 2022-03-04T16:45:40+08:00
draft: false
author: "Allen191819"
authorLink: "https://allen191819.xyz"
description: "Haskell I/O"

tags: ["Lambda", "FP"]
categories: ["Haskell"]

lightgallery: true

math:
    enable: true
resources:
- name: featured-image
  src: featured-image.jpg
---


输入与输出

<!--more-->

# Haskell Learning

## Functional Programming

- Pure functions
- Immutable Data
- No/Less side-effects
- Declatative
- Easier to verity

## 目录

<!-- vim-markdown-toc GFM -->

* [输入与输出](#输入与输出)
    * [运行一个 Haskell 程序](#运行一个-haskell-程序)
    * [Hello world](#hello-world)
    * [do block](#do-block)
    * [输入文本](#输入文本)
    * [其他 IO 相关函数用法](#其他-io-相关函数用法)
* [文件与流](#文件与流)
    * [openFile](#openfile)
    * [withFile](#withfile)
    * [readFile](#readfile)
    * [writeFile](#writefile)
    * [appendFile](#appendfile)
    * [buffer](#buffer)
    * [openTempFile](#opentempfile)
* [路径操作](#路径操作)
    * [getCurrentDirectory](#getcurrentdirectory)
    * [removeFile](#removefile)
    * [renameFile](#renamefile)
    * [doesFileExist](#doesfileexist)
* [Command line arguments](#command-line-arguments)
    * [getArgs](#getargs)
    * [getProgName](#getprogname)
* [Exceptions](#exceptions)

<!-- vim-markdown-toc -->

## 输入与输出

在 Haskell 中，一个函数不能改变状态，像是改变一个变量的内容。（当一个函数会改变状态，我们说这函数是有副作用的。） 函数无法改变状态的好处是它让我们促进了我们理解程序的容易度，但同时也造成了一个问题。假如说一个函数无法改变现实世界的状态，那它要如何打印出它所计算的结果？毕竟要告诉我们结果的话，它必须要改变输出设备的状态（譬如说屏幕），然后从屏幕传达到我们的脑，并改变我们心智的状态。

不要太早下结论，Haskell 实际上设计了一个非常聪明的系统来处理有副作用的函数，它漂亮地将我们的程序区分成纯粹跟非纯粹两部分。非纯粹的部分负责跟键盘还有屏幕沟通。有了这区分的机制，在跟外界沟通的同时，我们还是能够有效运用纯粹所带来的好处，像是惰性求值、容错性跟模块性。

### 运行一个 Haskell 程序

- 编译运行：

```bash
$ ghc --make helloworld
$ ./helloworld
```

- 使用 `runhaskell` 命令运行

```bash
$ runhaskell code.hs
```

### Hello world

在一个 Haskell 程序中输出文字需要定义一个 main 函数：

```haskell
main = putStrLn "Hello World"
```

其中 putStrLn 的类型是：

$$putStrLn :: String \to IO ()$$

`putStrLn` 接收一个 `String` 类型，并返回一个结果为()类型的 IO 动作（I/O action）。所以 main 函数的类型为 IO ()。（IO 的 Kind 是 _ -> _）

除此之外，还有其他默认提供的输出文本的函数：

- `putStr`：输出文本，结尾不换行
- `putChar`：输出单个字符，结尾不换行。接收的参数为单个 Char，不是 String（用单引号不是双引号）
- `print`：可以接收任何 Show 的成员，先用 show 转化为字符串然后输出。等同于 putStrLn . show

### do block

在 main 函数中使用多个 `putStrLn` 需要使用 `do` 语句：

```haskell
main = do
    putStrLn "晚宝晚宝 陪你到老"
    putStrLn "傲娇双钻 我的老伴"
```

其中最后一行一定要返回 `IO ()` 类型的值

### 输入文本

输入文字需要在 `do` 块中使用 `getLine`：

```haskell
main = do
line <- getLine
putStrLn line
```

`getLine` 的类型是：

$$getLine :: IO String$$

而 `<-` 操作符将 `getLine` 中的 String 提取了出来给到了 line，使 line 变成了 String 类型的一个字符串。

而且使用输入的字符串必须要经过一次 `<-`，不能直接使用 getLine 作为字符串，因为 getLine 不是 String 类型，而是 IO String 类型。

### 其他 IO 相关函数用法

**return**

Haskell 中的 `return` 和其他命令式语言中的 `return` 完全不同，它不会使函数直接结束并返回一个值。

main 函数必须定义为类型为 IO ()的函数，所以在 main 函数中使用 if 语句，如果不输出的话也不可以直接放下什么都不干，因为这时候 main 函数的类型不是 IO ()。所以这时需要使用 `return ()` 来为 main 函数指定为 IO ()类型，例如：

```haskell
main = do
line <- getLine
if null line
then return () -- <-这里
else do
...
```

使用 `<-` 操作符也可以直接将 `return` 语句中的内容提取出来，比如 `a <- return ‘A’`，执行后 a 就是’A’。

**when**

`when` 包含在 `Control.Monad` 模块中，它表示在满足第一个参数的条件下会执行第二个函数，否则会 `return ()`。比如：

```haskell
import Control.Monad

main = do
    c <- getChar
    when (c /= ' ') $ do
        putChar c
        main

```

等同于：

```haskell
main = do
    c <- getChar
    if c /= ' '
    then do
        putChar c
        main
    else return ()
```

**sequence**

`sequence` 在 IO 中使用时可以达成 `[IO a] -> IO [a]` 的效果，所以可以用作：

```haskell
[a, b, c] <- sequence [getLine, getLine, getLine]
```

**mapM & mapM\_**

在 IO 相关的地方使用 `map`，可以使用 `mapM` 和 `mapM_`，其中 `mapM` 有返回值而 `mapM_` 直接扔掉了返回值：

```haskell
ghci> mapM print [1,2,3]
1
2
3
[(),(),()]
ghci> mapM_ print [1,2,3]
1
2
3
```

**forever**

`forever` 函数包含在 `Control.Monad` 模块中。在 main 函数开头加上 `forever` 函数可以使后面的 do 块一直重复执行直到程序被迫终止，如：

```haskell
import Control.Monad

main = forever $ do
    ...
```

**forM**

`forM` 函数包含在 `Control.Monad` 模块中，它的功能和 `mapM` 类似，从第一个参数中逐个取出元素传入第二个参数（一个接收一个参数的函数）中，并且第二个参数可以返回 IO a 类型。比如：

```haskell
import Control.Monad

main = do
colors <- forM [1, 2, 3, 4] (\a -> do
putStrLn $ "Which color do you associate with the number " ++ show a ++ "?"
 color <- getLine
 return color)
putStrLn "The colors that you associate with 1, 2, 3 and 4 are: "
 mapM putStrLn colors

```

**getContents**

`getLine` 获取一整行，而 `getContents` 从标准输入中获取全部内容直到遇到 EOF，并且它是 lazy 的，在执行了 `foo <- getContents` 后，它并不会读取标准输入并且赋值到 foo，而是等到需要使用 foo 的时候再从标准输入读取。

`getContents` 在使用管道传入文字时很常用，可以代替 `forever` + `getLine` 使用，比如一个 Haskell 程序文件 code.hs：

```haskell
import Data.Char

main = do
    contents <- getContents
    putStr (map toUpper contents)
```

使用管道传入文本：

```bash
cat text.txt | ./code
```

**interact**

String -> String 类型的函数在输入输出中的使用太常见了，所以可以使用 interact 函数来简化。interact 的类型是：

$$interact :: (String \to String) \to IO ()$$

可以看出它接收一个 `String -> String` 的函数，并返回一个 IO ()类型，所以可以直接用在 main 上。

转换大写的程序就可以这样实现：

```haskell
main = interact $ unlines . map (map toUpper) . lines
```

## 文件与流

文件与流的相关函数都包含在 `System.IO` 模块中：

### openFile

$$openFile :: FilePath \to IOMode \to IO\ Handle$$

其中 `FilePath` 是 `String` 的 `type synonyms`，用一个字符串来表示需要打开的文件的路径

`IOMode` 的定义是：

```haskell
data IOMode = ReadMode | WriteMode | AppendMode | ReadWriteMode
```

`openFile` 返回一个 IO Handle 类型的值，将其用<-操作符提取后会出现一个 Handle 的值。但不能从 Handle 中直接使用文字，还需要使用一系列函数：

`hGetContents` :: Handle -> IO String ，从 Handle 中读取全部内容，返回一个 IO String
`hGetChar` :: Handle -> IO Char ，从 Handle 中读取一个字符
`hGetLine` :: Handle -> IO String ，从 Handle 中读取一行，返回一个 IO String
`hPutStr` :: Handle -> String -> IO () ，向 Handle 中输出字符串
`hPutStrLn` :: Handle -> String -> IO () ，同上

在使用 `openFile` 进行文件操作后，需要使用 `hClose` 手动关闭 Handle。

$$hClose :: Handle \to IO ()$$

接收一个 Handle 并返回 IO ()，可以直接放在 main 函数末尾

所以使用 openFile 读取一个文件中的全部内容并输出的全部代码是：

```haskell
import System.IO

main = do
    handle <- openFile "test.txt" ReadMode
    contents <- hGetContents handle
    putStrLn contents
    hClose handle
```

### withFile

`withFile` 类似 Python 中的 `with open`，它在读取文件使用之后不需要手动 `close` 文件。它的类型是：

$$withFile :: FilePath \to IOMode \to (Handle \to IO a) \to IO a$$

可以看出，它接收三个参数：

- `FilePath`：一个表示文件路径的 String

- `IOMode`：打开文件的模式

- `(Handle -> IO a)`：一个函数，表示对读取文件后的 Handle 索要进行的操作，需要返回一个 I/O action；而这个返回值也将作为 withFile 的返回值

```haskell
import System.IO

main = withFile "text.txt" ReadMode (\handle -> do
    contents <- hGetContents handle
    putStrLn contents)
```

### readFile

`readFile` 可以更加简化读取文件内容的操作，它的类型：

$$readFile :: FilePath \to IO String$$

它只需要输入一个表示文件路径的字符串，返回其中以其中内容为内容的 `I/O action`：

```haskell
import System.IO

main = do
    contents <- readFile "text.txt"
    putStrLn contents
```

### writeFile

writeFile 简化了写入文件的操作，它的类型：

$$writeFile :: FilePath \to String \to IO ()$$

传入的第一个参数是要写入的文件的路径，第二个参数是要写入的字符串，返回一个 IO ()

### appendFile

`appendFile` 类似 `writeFile`，但使用它不会覆盖文件中原来内容，而是直接把字符串添加到文件末尾

### buffer

文件以流的形式被读取，默认文字文件的缓冲区 `buffer` 大小是一行，即每次读取一行内容；默认二进制文件的缓冲区大小是以块为单位，如果没有指定则根据系统默认来选择。

也可以通过 `hSetBuffering` 函数来手动设置缓冲区大小，这个函数的类型：

$$hSetBuffering :: Handle \to BufferMode \to IO ()$$

它接收一个 `handle`，和一个 `BufferMode`，并返回 IO ()。其中 `BufferMode` 有以下几种：

- `NoBuffering`：没有缓冲区，一次读入一个字符
- `LineBuffering`：缓冲区大小是一行，即每次读入一行内容
- `BlockBuffering (Maybe Int)`：缓冲区大小是一块，块的大小由 Maybe Int 指定：
  - `BlockBuffering (Nothing)`：使用系统默认的块大小
  - `BlockBuffering (Just 2048)`：一块的大小是 2048 字节，即每次读入 2048bytes 的内容

缓冲区的刷新是自动的，也可以通过 `hFlush` 来手动刷新

$$hFlush :: Handle \to IO ()$$

传入一个 handle，返回 IO ()，即刷新对应 handle 的缓冲区

### openTempFile

`openTempFile` 可以新建一个临时文件：

$$openTempFile :: FilePath \to String \to IO (FilePath, Handle)$$

`FilePath` 指临时文件要创建的位置路径，`String` 指临时文件名字的前缀，返回一个 `I/O action`，其内容第一个 FilePath 是创建得到的临时文件的路径，Handle 是临时文件的 handle

例如：

```haskell
import System.IO

main = do
    (tempFile, tempHandle) <- openTempFile "." "temp"
    ...
    hClose tempHandle
```

## 路径操作

相关函数都包含在 `System.Directory` 模块中.

### getCurrentDirectory

$$getCurrentDirectory :: IO FilePath$$

直接返回一个 I/O action，其内容是一个字符串表示当前路径的绝对路径

### removeFile

$$removeFile :: FilePath \to IO ()$$

输入一个文件路径，并删除掉它

### renameFile

$$renameFile :: FilePath \to FilePath \to IO ()$$

输入一个原路径，一个新路径，为原路径的文件重命名为新路径的名

### doesFileExist

$$doesFileExist :: FilePath \to IO Bool$$

检查文件是否存在，返回一个包含布尔值的 I/O action

## Command line arguments

`System.Environment` 模块中提供了两个函数可以用来处理传入命令行的参数

### getArgs

$$getArgs :: IO [String]$$

不需要输入参数，直接返回一个 `I/O action`，内容为传入命令行的参数（一个由 String 组成的列表）。相当于 C 语言中的 argv[1:]

### getProgName

$$getProgName :: IO String$$

返回 I/O action，内容为程序的名字，相当于 C 语言中的 argv[0]

## Exceptions

程序在运行失败时会抛出异常，可以通过 `Control.Exception` 模块中的 catch 函数来捕获异常：

$$catch :: Exception e \Rightarrow IO a \to (e \to IO a) \to IO a$$

第一个参数是要进行的操作，以 IO a 为返回值的类型，第二个参数是一个函数，它接收异常并进行操作，例如：

```haskell
import Control.Exception

main = main' `catch` handler

main' :: IO ()
main' = do
    ...

handler :: Exception e => e -> IO ()
handler e =  putStrLn "..."
```

也可以利用 `guard` 语法和 `System.IO.Error` 中的函数来判断 IO 异常的类型来进行不同操作：

```haskell
import System.Environment
import System.IO.Error
import Control.Exception

main = toTry `catch` handler

toTry :: IO ()

toTry = do (fileName:_) <- getArgs

           contents <- readFile fileName

           putStrLn $ "The file has " ++ show (length (lines contents)) ++ " lines!"



handler :: IOError -> IO ()

handler e

    | isDoesNotExistError e = putStrLn "The file doesn't exist!"

    | otherwise = ioError e

```
