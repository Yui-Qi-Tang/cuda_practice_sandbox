# 本來想試試在玩windows環境上跑個cuda專案，但實在坑太多，我就不碰了，專心在LINUX就好。

# 幹你媽我真快抓狂(以下是血淚史)

    先是去你媽的nvcc on win10 需要 他媽的 vs cl.exe, 我她媽找老半天, 裝了vs studio然後cl.exe放的位置亂七八糟!!

    又他媽的一堆資料夾 三小 host_x86, host_x64 底下都有cl.exe，幹你娘

    好不容易可以編譯了，nvcc又幹你媽的回報不支持vs studio 2017???幹你娘我就是裝vs studio 2017阿?

    後來找到要在 要在幹你媽的 C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\include\crt\host_config.h:133 底下修改版本號

    我懶得考察幹你媽的vs studio的版本是他媽的幾年生的 我就去她嗎的網路查

    一開始查到是 **#if _MSC_VER < 1600 || _MSC_VER > 1911**

    然後去你媽的還是編不過，最後就+1+1+1+1/-1-1-1-1-1-1，幹你媽的最後是:
    
    **#if _MSC_VER < 1600 || _MSC_VER > 1920**

    幹你娘機掰 cuda on win10 and windows fuckyou!!!!

    阿幹你娘還有!

    nvcc 需要 vs studio 的compiler，然後windows平台上的utf8編碼都需要在檔案前加上BOM!!!!!

    所以你媽的程式碼用utf8編碼送給幹你娘的vs studio編譯器吃，他會跟你他說 **幹你娘機掰你的檔案不是utf8編碼!!!!**
