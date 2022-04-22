---
weight: 4
title: "Haskell 学习笔记一"
date: 2022-03-02T21:57:40+08:00
lastmod: 2020-03-02T16:45:40+08:00
draft: false
author: "Allen191819"
authorLink: "https://allen191819.xyz"
description: "Haskell 基础语法"

tags: ["Lambda", "FP"]
categories: ["Haskell"]

lightgallery: true

math:
  enable: true
resources:
  - name: featured-image
    src: featured-image.png
---

Haskell 基础语法

<!--more-->

# Haskell Learning

## Functional Programming

- Pure functions
- Immutable Data
- No/Less side-effects
- Declatative
- Easier to verity

## 基础语法

### 基础运算

| Char          | Function        |
| ------------- | --------------- |
| `+ - \* /`    | 加减乘除        |
| `div`         | 整除            |
| `mod`         | 取模            |
| `True Flase`  | Boolean         |
| `\|\| && not` | 或且非          |
| `==`          | 条件判断 相等   |
| `\\=`         | 条件判断 不相等 |

### 函数调用

```haskell
ghci> max 1 2
2
```

中缀函数与前缀函数的转换(`prefix` & `infix`)

- 对前缀函数加<code>``</code>使其变成中缀函数
- 对中缀函数加`()`使其变成前缀函数

```haskell
ghci> 4 `div` 2
2
ghci> 1 `max` 2
2
ghci> (+) 1 2
3
ghci> (||) True False
True
```

### List

**List 常用函数**

- `(++)` :: [a] -> [a] -> [a]：合并两个列表

- `(:)` :: a -> [a] -> [a]：将单个元素并入列表。[1, 2, 3]是 1:2:3:[]的语法糖

- `(!!)` :: [a] -> Int -> a：通过索引取出某个位置上的元素。a !! 1 相当于 Python 中的 a[1]

- `head` :: [a] -> a：返回列表的第一个元素

- `tail` :: [a] -> [a]：返回列表中除去第一个元素后的列表（若只有一个元素则返回空列表[ ]）

- `last` :: [a] -> a：返回列表中的最后一个元素

- `init` :: [a] -> [a]：返回列表中除去最后一个元素后的列表

- `length` :: Foldable t => t a -> Int：返回列表的长度

- `null` :: Foldable t => t a -> Bool：返回列表是否为空

- `reverse` :: [a] -> [a]：返回翻转后的列表

- `take` :: Int -> [a] -> [a]：返回列表 a 的前 n 个元素的列表(take n a)

- `drop` :: Int -> [a] -> [a]：返回列表 a 中除去前 n 个元素后的列表(drop n a)

- `maximum` :: (Foldable t, Ord a) => t a -> a：返回列表中的最大值

- `minimum` :: (Foldable t, Ord a) => t a -> a：返回列表中的最小值

- `sum` :: (Foldable t, Num a) => t a -> a：返回列表中所有元素的和

- `product` :: (Foldable t, Num a) => t a -> a：返回列表中所有元素的积

- `elem` :: (Foldable t, Eq a) => t a -> Bool：判断值 n 是否在列表 a 中

**Range**: `..`

```haskell
ghci> [1 .. 10]
[1,2,3,4,5,6,7,8,9,10]
ghci> ['a' .. 'z']
"abcdefghijklmnopqrstuvwxyz"
ghci> ['K' .. 'Z']
"KLMNOPQRSTUVWXYZ"
ghci> [2, 4 .. 20]
[2,4,6,8,10,12,14,16,18,20]
ghci> [3, 6 .. 20]
[3,6,9,12,15,18]
ghci> [5, 4 .. 1]
[5,4,3,2,1]
```

haskell 是惰性的，生成无穷列表之后通过 `take` 生成 list:

- `cycle` :: [a] -> [a]：将原列表不断循环生成无穷列表

- `repeat` :: a -> [a]：将传入的值不断重复生成无穷列表

  - `replicate` :: Int -> a -> [a]：将值 a 重复 n 次，返回生成的列表(replicate n a)

### List comprehension

```haskell
ghci> [x * 2 | x <- [1 .. 10]]
[2,4,6,8,10,12,14,16,18,20]
ghci> [x * 2 | x <- [1 .. 10], x * 2 >= 12]
[12,14,16,18,20]
ghci> [ x | x <- [50 .. 100], x `mod` 7 == 3]
[52,59,66,73,80,87,94]
ghci> [x * y | x <- [2, 5, 10], y <- [8, 10, 11]]
[16,20,22,40,50,55,80,100,110]
```

### Tuple

Haskell 中的元组可以有不同的长度，元素类型也可以不同，元组类型由其中的所有元素的类型共同决定。二元元组的常用函数：

- `fst` :: (a, b) -> a：返回含有两个元素元组中的第一个元素
- `snd` :: (a, b) -> b：返回含有两个元素元组中的第二个元素
- `zip` :: [a] -> [b] -> [(a, b)]：接收两个列表，返回一个列表，每个元素是依次将两个列表中元素配对成的二元组

## Syntax in Functions

### 定义函数

直接定义一个函数：

```haskell
add' x y = x + y
```

这时 Haskell 会自动推断函数的类型为(Num a) => a -> a -> a。但是最好在定义函数前声明函数的类型：

```haskell
add' :: (Num a)=>a->a->a
add' x y = x + y
```

### Pattern matching

```haskell
luckyNumber :: (Integral a)=>a->String
luckyNumber 6 = "Lucky number six!"
luckyNumber x = "Sorry, you're out of luck."
```

注意：在定义模式时，一定要留一个万能匹配的模式，这样我们的进程就不会为了不可预料的输入而崩溃了。

(x:xs) 模式

```haskell
sum' :: (Num a) => [a] -> a
sum' [] = 0
sum' (x:xs) = x + sum' xs
```

as 模式:将一个名字和 @ 置于模式前，可以在按模式分割什么东西时仍保留对其整体的引用。

```haskell
capital :: String -> String
capital "" = "Empty string, whoops!"
capital all@(x:xs) = "The first letter of " ++ all ++ " is " ++ [x]
```

### Guards

```haskell
Compare :: (Ord a) => a -> a -> Ordering
a `Compare` b
    | a > b     = GT
    | a == b    = EQ
    | otherwise = LT
```

### 关键字 where 和 let

```haskell
bmiTell :: (RealFloat a) => a -> a -> String
bmiTell weight height
    | bmi <= 18.5 = "You're underweight"
    | bmi <= 25.0 = "You're supposedly normal."
    | bmi <= 30.0 = "You're fat!"
    | otherwise   = "You're a whale!"
    where bmi = weight / height ^ 2
```

```haskell
quicksort :: (Ord a) => [a] -> [a]
quicksort [] = []
quicksort (x : xs) =
  let smallerSorted = quicksort [a | a <- xs, a <= x]
      biggerSorted = quicksort [a | a <- xs, a > x]
   in smallerSorted ++ [x] ++ biggerSorted
```

### Case expressions

```haskell
describeList :: [a] -> String
describeList xs = "The list is " ++ case xs of [] -> "empty."
                                               [x] -> "a singleton list."
                                               xs -> "a longer list."
```

```haskell
case expression of pattern -> result
                   pattern -> result
                   pattern -> result
                   ...
```
