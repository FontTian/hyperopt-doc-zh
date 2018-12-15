# [安装说明](https://github.com/hyperopt/hyperopt/wiki/Installation-Notes)
*hyperopt安装说明*
*[Font Tian](http://blog.csdn.net/fontthrone) translated this article on 23 December 2017*

## 有关MongoDB的部分
Hyperopt要求[mongodb](http://www.mongodb.org/)（有时候简称“mongo”）来执行并行搜索。据我所知，hyperopt与2.xx系列中的所有版本兼容，这是目前的（[在这里下载最新版本](http://www.mongodb.org/downloads)）。它甚至可能与mongodb的所有版本兼容，我不知道mongo的任何特定的版本要求。

在linux和OSX上，一旦你下载了mongodb并解压，只需将它链接到bin/你的virtualenv 的子目录中，然后安装完成。

```
# from the root of your virtualenv
# (or basically any folder with an active bin/ subdirectory)
(cd bin && { for F in ../mongodb-linux-x86_64-2.2.2/bin/* ; do echo "linking $F" ; ln -s $F ; done } )

```
通过运行完整的单元测试套件或者只是mongo文件来验证hyperopt是否可以使用mongodb

```
# cd to the hyperopt project root
nosetests hyperopt/tests/test_mongoexp.py
```