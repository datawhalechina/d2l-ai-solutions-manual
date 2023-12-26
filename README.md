# 动手学深度学习习题解答
&emsp;&emsp;李沐老师的[《动手学深度学习》](https://zh-v2.d2l.ai/index.html#)是入门深度学习的经典书籍，这本书基于深度学习框架来介绍深度学习，书中代码可以做到“所学即所用”。对于一般的初学者来说想要把书中课后习题部分独立解答还是比较困难。本项目对《动手学深度学习》习题部分进行解答，作为该书的习题手册，帮助初学者快速理解书中内容。

## 使用说明
&emsp;&emsp;动手学深度学习习题解答，主要完成了该书的所有习题，并提供代码和运行之后的截图，里面的内容是以深度学习的内容为前置知识，该习题解答的最佳使用方法是以李沐老师的《动手学深度学习》为主线，并尝试完成课后习题，如果遇到不会的，再来查阅习题解答。  
&emsp;&emsp;如果觉得解答不详细，可以[点击这里](https://github.com/datawhalechina/d2l-ai-solutions-manual/issues)提交你希望补充推导或者习题编号，我们看到后会尽快进行补充。

### 在线阅读地址
在线阅读地址：https://datawhalechina.github.io/d2l-ai-solutions-manual

## 选用的《动手学深度学习》版本

<div align=center>
   <img src="images/book.png?raw=true" width="336" height= "500">
</div>

> 书名：动手学深度学习（PyTorch版）  
> 著者：阿斯顿·张、[美]扎卡里 C. 立顿、李沐、[德]亚历山大·J.斯莫拉  
> 译者：何孝霆、瑞潮儿·胡  
> 出版社：人民邮电出版社    
> 版次：2023年2月第1版  

## Notebook运行环境配置

1. 克隆项目请使用如下命令(只克隆最新的 commit )：

   ```shell
   git clone https://github.com/datawhalechina/d2l-ai-solutions-manual.git --depth 1
   ```

2. Python版本  
   请使用python3.10.X，如使用其他版本，requirements.txt中所列的依赖包可能不兼容。
   
3. 安装相关的依赖包


    ```shell
    pip install -r requirements.txt
    ```

4. 安装PyTorch

   访问[PyTorch官网](https://pytorch.org/get-started/locally/)，选择合适的版本安装PyTorch，有条件的小伙伴可以下载GPU版本
   ```shell
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

5. 安装d2l
   ```shell
   pip install d2l
   ```

6. docsify框架运行
    ```shell
    docsify serve ./docs
    ```

## 协作规范

1. 由于习题解答中需要有程序和执行结果，采用jupyter notebook的格式进行编写（文件路径：notebooks），然后将其导出成markdown格式，再覆盖到docs对应的章节下。
2. 可按照Notebook运行环境配置，配置相关的运行环境。
3. 习题解答编写中，需要尽量使用初学者（有高数基础）能理解的数学概念，如果涉及公式定理的推导和证明，可附上参考链接。
4. 当前进度

<div align=center>

| 章节号 |           标题            |  进度   | 负责人  | 审核人  |
| :---: | :----------------------: | :----: | :----: | :----: |
|  2    | 预备知识                   | 待审核 | 毛瑞盈、陈可为、胡锐锋 |  |
|  3    | 线性神经网络                | 待审核 | 毛瑞盈、陈可为、胡锐锋 |  |
|  4    | 多层感知机                  | 待审核 | 毛瑞盈、陈可为 |  |
|  5    | 深度学习计算                | 待审核 | 宋志学、韩颐堃 |   |
|  6    | 卷积神经网络                | 待审核 | 宋志学、韩颐堃 |  |
|  7    | 现代卷积神经网络             | 待审核 | 宋志学、韩颐堃 |  |
|  8    | 循环神经网络                | 待审核 | 王振凯 |  |
|  9    | 现代循环神经网络             | 待审核 | 王振凯 |  |
| 10    | 注意力机制                  | 待审核 | 徐韵婉、崔腾松 |  |
| 11    | 优化算法                    | 待审核 | 张银晗、邹雨衡 |  |
| 12    | 计算性能                    | 待审核 | 邹雨衡 |  |
| 13    | 计算机视觉                  | 待审核 | 刘旭、曾莹 |  |
| 14    | 自然语言处理：预训练          | 待审核 | 肖鸿儒 |  |
| 15    | 自然语言处理：应用            | 待审核 | 张友东、张凯旋 |  |

</div>

## 项目结构
<pre>
codes----------------------------------------------习题代码
docs-----------------------------------------------习题解答
notebook-------------------------------------------习题解答JupyterNotebook格式
requirements.txt-----------------------------------运行环境依赖包
</pre>
## 致谢

**核心贡献者**

- [宋志学-项目负责人](https://github.com/KMnO4-zx)（Datawhale成员-河南理工大学）
- [胡锐锋-项目发起人](https://github.com/Relph1119)（Datawhale成员-华东交通大学-系统架构设计师）
- [韩颐堃](https://github.com/YikunHan42)（内容创作者-Datawhale成员）
- [毛瑞盈](https://github.com/catcooc)（内容创作者）
- [陈可为](https://github.com/Ethan-Chen-plus)（内容创作者-Datawhale成员）
- [王振凯](https://github.com/Rookie-Kai)（内容创作者-Datawhale成员）
- [崔腾松](https://github.com/2951121599)（Datawhale成员-whale量化开源项目负责人）
- [徐韵婉](https://github.com/xyw1)（内容创作者-Datawhale成员）
- [张银晗](https://github.com/YinHan-Zhang)（内容创作者-Datawhale成员）
- [邹雨衡](https://github.com/logan-zou)（Datawhale成员-对外经济贸易大学研究生）
- [刘旭](https://github.com/liuxu-manifold)（内容创作者-Datawhale成员-深圳大学）
- [曾莹](https://github.com/zengying321)（内容创作者-深圳大学）
- [肖鸿儒](https://github.com/Hongru0306)（内容创作者-同济大学）
- [张友东](https://github.com/AXYZdong)（内容创作者-Datawhale成员）
- [张凯旋](https://github.com/zarjun)（内容创作者-Datawhale成员-上海科技大学）

**其他**
- 特别感谢[@Sm1les](https://github.com/Sm1les)对本项目的帮助与支持
- 如果有任何想法可以联系我们 DataWhale 也欢迎大家多多提出 issue
- 特别感谢以下为教程做出贡献的同学！

<a href="https://github.com/datawhalechina/d2l-ai-solutions-manual/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=datawhalechina/d2l-ai-solutions-manual" />
</a>

## 参考文献
- [动手学深度学习-中文版](https://zh.d2l.ai/)
- [动手学深度学习-英文版](https://d2l.ai/)

## Star History

![Star History Chart](https://api.star-history.com/svg?repos=datawhalechina/d2l-ai-solutions-manual&type=Date)

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号：Datawhale</p>
<img src="images/qrcode.jpeg" width = "180" height = "180">
</div>
&emsp;&emsp;Datawhale，一个专注于AI领域的学习圈子。初衷是for the learner，和学习者一起成长。目前加入学习社群的人数已经数千人，组织了机器学习，深度学习，数据分析，数据挖掘，爬虫，编程，统计学，Mysql，数据竞赛等多个领域的内容学习，微信搜索公众号Datawhale可以加入我们。

## LICENSE
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。