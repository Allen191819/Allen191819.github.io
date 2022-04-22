# Haskell 学习笔记二


高阶函数与模块

<!--more-->

# Haskell Learning

## 高阶函数

### Curried function

Haskell 中的所有函数都只有一个参数，所有多参数函数都是 **Curried
function**(柯里化的函数)，例如一个二元函数 f x y，执行时，他首先会回传一个取一个参数的函数
f x， 再用参数 y 调用它。

$$max :: Ord a \Rightarrow a \to a \to a$$

```haskell
ghci> max 4 5
5
ghci> (max 4) 5
5
```

### 高阶函数

Haskell 中的函数可以取另一个函数作为参数，也可以传回函数。

```haskell
applyTwice ::(a->a) -> a -> a
applyTwice f x = f (f x)
```

该函数会连续调用两次 f 函数

### 一些高阶函数

**zipWith**

$$zipWith :: (a \to b \to c) \to [a] \to [b] \to [c]$$

第一个参数为一个函数，然后接收两个列表，将其对应元素传入接收的函数中，得到的结果组成一个新的列表。如果两个传入的列表长度不同，以最短的列表为准，长列表中超出的元素省略。用例：

```haskell
ghci> zipWith (*) [1,4,3,7] [1,7,3,4,7]
[1,28,9,28]
```

**flip**

$$flip :: (a \to b \to c) \to b \to a \to c$$

flip 函数接收一个二元函数，返回一个新的二元函数，将其输入的两个参数顺序反过来：

```haskell
ghci> zip [1,2,3,4,5] "hello"
[(1,'h'),(2,'e'),(3,'l'),(4,'l'),(5,'o')]
ghci> flip zip [1,2,3,4,5] "hello"
[('h',1),('e',2),('l',3),('l',4),('o',5)]
```

**map**

$$map :: (a \to b) \to [a] \to [b]$$

map 函数接收一个函数 f 和一个列表 a，将函数 f 应用在列表 a 的每个元素中，并返回得到的所有结果组成的列表 b：

```haskell
map (max 6) [1,3,4,9,12]
[6,6,6,9,12]
```

**filter**

$$filter :: (a \to Bool) \to [a] \to [a]$$

filter 函数接收一个函数 f 和一个列表 a，将列表 a 中的每个元素传入函数 f 中，如果结果为 True 就保留，结果为 False 就抛弃，返回所有保留的元素组成的新列表：

```haskell
ghci> filter even [1..10]
[2,4,6,8,10]
```

**takeWhile**

$$takeWhile :: (a \to Bool) \to [a] \to [a]$$

takeWhile 函数接收一个函数 f 和一个列表 a，将列表 a 中从左向右每个元素传入函数 f，直到结果为 False 停止，返回停止前传入的所有元素组成的新列表：

```haskell
ghci> takeWhile (/=' ') "word1 word2"
"word1"
```

### Function application

函数应用可以使用`$`，`$`是一个函数，它的类型是：

$$($) :: (a \to b) \to a \to b$$

它可以改变函数结合优先级，将左侧函数应用于全部右侧内容上，相当于给右侧整体加上了括号。否则函数默认左结合，会依次向右应用而不会应用在整体上。

```haskell
f $ g x
-- 等价于
f (g x)
-----
f g x
-- 等价于
(f g) x
```

### Function Composition

函数复合可以使用`.`，`.`也是一个函数，它的类型是：

$$(.) :: (b \to c) \to (a \to b) \to a \to c$$

定义是：

$$
f . g = \\x \to f (g x)
$$

但是函数复合的优先级要比函数执行低，比如：

```haskell
sum . replicate 5 . max 6.7 8.9
```

会先执行 max 6.7 8.9 并返回 8.9，然后将 sum、replicate 5、8.9 复合，但两个函数无法和一个值(8.9)复合，所以会抛出异常。因此要使用$来规定先复合再执行：

```haskell
sum . replicate 5 . max 6.7 $ 8.9
```

### lambda λ

lambda 就是匿名函数。有些时候我们需要传给高阶函数一个函数，而这函数我们只会用这一次，这就弄个特定功能的 lambda。编写 lambda，就写个 \ (因为它看起来像是希腊字母的 λ ? 其实我觉得一点都不像)，后面是用空格分隔的参数，$\to$ 后面就是函数体。通常我们都是用括号将其括起，要不然它就会占据整个右边部分。

```haskell
\para1 para2 ... -> return
```

## Modules

Haskell 会自动加载 Prelude 模块（module），如果在 GHCi 中再加载其他模块，需要使用:m + ...，比如加载 Data.List 模块：

```haskell
Prelude> :m + Data.List
```

而在 hs 文件中引入模块，需要使用 import 语句:

```haskell
import Data.List

import Data.List (nub, sort)            -- 仅 import nub sort 函数

import Data.List hiding (nub)           -- 不 improt nub 函数

import qualified Data.List

import qualified Data.List as Li
```

### 编写自己的 Modules

模块中要包含将要使用的一些函数，像正常的 hs 文件一样写即可，但头部需要有导出语句（export）。比如一个模块文件名叫`Geometry.hs`：

```haskell
module Geometry
( sphereVolume
, sphereArea
, cubeVolume
, cubeArea
, cuboidArea
, cuboidVolume
) where

sphereVolume :: Float -> Float
sphereVolume radius = (4.0 / 3.0) * pi * (radius ^ 3)

sphereArea :: Float -> Float
sphereArea radius = 4 * pi * (radius ^ 2)

cubeVolume :: Float -> Float
cubeVolume side = cuboidVolume side side side

cubeArea :: Float -> Float
cubeArea side = cuboidArea side side side

cuboidVolume :: Float -> Float -> Float -> Float
cuboidVolume a b c = rectangleArea a b * c

cuboidArea :: Float -> Float -> Float -> Float
cuboidArea a b c = rectangleArea a b * 2 + rectangleArea a c * 2 + rectangleArea c b * 2

rectangleArea :: Float -> Float -> Float
rectangleArea a b = a * b
```

在调用该模块时，只能调用`module Geometry(...)`
中包含的内容，其他不在其中的函数都不调用。

