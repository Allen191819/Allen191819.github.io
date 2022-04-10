---
weight: 4
title: "Markdown 嵌入关系图表"
date: 2021-12-04T21:57:40+08:00
lastmod: 2021-12-06T16:45:40+08:00
draft: false
author: "Allen191819"
authorLink: "https://allen191819.xyz"
description: "对 mermaid 的介绍"

tags: ["Daily study", "Markdown"]
categories: ["Markdown"]

lightgallery: true

math:
    enable: true
resources:
- name: featured-image
  src: featured-image.jpg
---

借助 mermaid 在 markdown 中嵌入图表

<!--more-->

# Mermaid learn

{{< figure src="header.png" size=300x300 >}}

## Mermaid 可以绘制的图表类型

-   饼状图：使用 pie 关键字，具体用法后文将详细介绍
-   流程图：使用 graph 关键字，具体用法后文将详细介绍
-   序列图：使用 sequenceDiagram 关键字
-   甘特图：使用 gantt 关键字
-   类图：使用 classDiagram 关键字
-   状态图：使用 stateDiagram 关键字
-   用户旅程图：使用 journey 关键字

*   一个例子`Archlinux yyds`

```
graph TD
    A[Choosing an OS] --> B{Do you fear technology ?}
    B -->|Yes| C{Is your daddy rich ?}
    C -->|Yes| E(fa:fa-apple Mac OS);
    C -->|No| F(fa:fa-chrome Chrome OS);

    B -->|No| D{Do you care about freedom/privacy ?}
    D -->|Yes| G{Do you have a life ?}
    D -->|No| H(fa:fa-windows Windows);

    G -->|Yes| I(fa:fa-ubuntu Ubuntu);
    G -->|Yes| K(fa:fa-fedora Fedora);
    G -->|No| L(fa:fa-linux Archlinux);
    G -->|No| M(fa:fa-shield Backtrack);

    style A fill:#0094FF,stroke:#333,stroke-width:4px,color:#fff
    style E fill:#808080,stroke:#333,stroke-width:2px,color:#fff,stroke-dasharray: 5 5
    style F fill:#808080,stroke:#333,stroke-width:2px,color:#fff,stroke-dasharray: 5 5
    style H fill:#004A7F,stroke:#333,stroke-width:2px,color:#fff,stroke-dasharray: 5 5
    style I fill:#FF6A00,stroke:#333,stroke-width:2px,color:#fff,stroke-dasharray: 5 5
    style K fill:#FF6A00,stroke:#333,stroke-width:2px,color:#fff,stroke-dasharray: 5 5
    style L fill:#7F0000,stroke:#333,stroke-width:2px,color:#fff,stroke-dasharray: 5 5
    style M fill:#7F0000,stroke:#333,stroke-width:2px,color:#fff,stroke-dasharray: 5 5
```

{{< mermaid >}}
graph TD
A[Choosing an OS] --> B{Do you fear technology ?}
B -->|Yes| C{Is your daddy rich ?}
C -->|Yes| E(fa:fa-apple Mac OS);
C -->|No| F(fa:fa-chrome Chrome OS);

    B -->|No| D{Do you care about freedom/privacy ?}
    D -->|Yes| G{Do you have a life ?}
    D -->|No| H(fa:fa-windows Windows);

    G -->|Yes| I(fa:fa-ubuntu Ubuntu);
    G -->|Yes| K(fa:fa-fedora Fedora);
    G -->|No| L(fa:fa-linux Archlinux);
    G -->|No| M(fa:fa-shield Backtrack);

    style A fill:#0094FF,stroke:#333,stroke-width:4px,color:#fff
    style E fill:#808080,stroke:#333,stroke-width:2px,color:#fff,stroke-dasharray: 5 5
    style F fill:#808080,stroke:#333,stroke-width:2px,color:#fff,stroke-dasharray: 5 5
    style H fill:#004A7F,stroke:#333,stroke-width:2px,color:#fff,stroke-dasharray: 5 5
    style I fill:#FF6A00,stroke:#333,stroke-width:2px,color:#fff,stroke-dasharray: 5 5
    style K fill:#FF6A00,stroke:#333,stroke-width:2px,color:#fff,stroke-dasharray: 5 5
    style L fill:#7F0000,stroke:#333,stroke-width:2px,color:#fff,stroke-dasharray: 5 5
    style M fill:#7F0000,stroke:#333,stroke-width:2px,color:#fff,stroke-dasharray: 5 5

{{< /mermaid >}}

## 流程图

```
graph TD
A[Hard] -->|Text| B(Round)
B --> C{Decision}
C -->|One| D[Result 1]
C -->|Two| E[Result 2]
```

[doc](https://mermaid-js.github.io/mermaid/#/flowchart)

{{< mermaid >}}
graph TD
A[Hard] -->|Text| B(Round)
B --> C{Decision}
C -->|One| D[Result 1]
C -->|Two| E[Result 2]
{{< /mermaid >}}

## 时序图

```
sequenceDiagram
Alice->>John: Hello John, how are you?
loop Healthcheck
    John->>John: Fight against hypochondria
end
Note right of John: Rational thoughts!
John-->>Alice: Great!
John->>Bob: How about you?
Bob-->>John: Jolly good!
```

[doc](https://mermaid-js.github.io/mermaid/#/sequenceDiagram)
{{< mermaid >}}
sequenceDiagram
Alice->>John: Hello John, how are you?
loop Healthcheck
John->>John: Fight against hypochondria
end
Note right of John: Rational thoughts!
John-->>Alice: Great!
John->>Bob: How about you?
Bob-->>John: Jolly good!
{{< /mermaid >}}

## 甘特图

```
gantt
section Section
Completed :done,    des1, 2014-01-06,2014-01-08
Active        :active,  des2, 2014-01-07, 3d
Parallel 1   :         des3, after des1, 1d
Parallel 2   :         des4, after des1, 1d
Parallel 3   :         des5, after des3, 1d
Parallel 4   :         des6, after des4, 1d
```

[doc](https://mermaid-js.github.io/mermaid/#/gantt)

{{< mermaid >}}
gantt
section Section
Completed :done, des1, 2014-01-06,2014-01-08
Active :active, des2, 2014-01-07, 3d
Parallel 1 : des3, after des1, 1d
Parallel 2 : des4, after des1, 1d
Parallel 3 : des5, after des3, 1d
Parallel 4 : des6, after des4, 1d
{{< /mermaid >}}

## 类图

```
classDiagram
Animal <|-- Duck
Animal <|-- Fish
Animal <|-- Zebra
Animal : +int age
Animal : +String gender
Animal: +isMammal()
Animal: +mate()
class Duck{
+String beakColor
+swim()
+quack()
}
class Fish{
-int sizeInFeet
-canEat()
}
class Zebra{
+bool is_wild
+run()
}
```

[doc](https://mermaid-js.github.io/mermaid/#/classDiagram)

{{< mermaid >}}
classDiagram
Animal <|-- Duck
Animal <|-- Fish
Animal <|-- Zebra
Animal : +int age
Animal : +String gender
Animal: +isMammal()
Animal: +mate()
class Duck{
+String beakColor
+swim()
+quack()
}
class Fish{
-int sizeInFeet
-canEat()
}
class Zebra{
+bool is_wild
+run()
}
{{< /mermaid >}}

## 状态图

```
stateDiagram-v2
    [*] --> Still
    Still --> [*]

    Still --> Moving
    Moving --> Still
    Moving --> Crash
    Crash --> [*]
```

[doc](https://mermaid-js.github.io/mermaid/#/stateDiagram)
{{<mermaid>}}
stateDiagram-v2
[*] --> Still
Still --> [*]

    Still --> Moving
    Moving --> Still
    Moving --> Crash
    Crash --> [*]

{{</mermaid>}}

## 饼状图

```
pie title 为什么不喜欢出门
    "懒" : 50
    "社恐" : 40
    "穷" : 200
```

{{< mermaid >}}
pie title 为什么不喜欢出门
"懒" : 50
"社恐" : 40
"穷" : 200
{{< /mermaid >}}
