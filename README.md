# 先前太多情緒字眼，特改一版本，舊版本就當紀錄，當個笑話...

# 這裡紀錄一下在Windows開發CUDA的困擾
    
    開發版本是visual studio 2017 + CUDA 9.2

    1. windows vc的c/c++ compiler是cl.exe, 下載時得注意那個擺在哪裡（好像visual studio 版本不同放的也不同？）; 若想省找編譯器的步驟，可以執行環境設定那類的（vxsetup_x86??具體檔名我忘了），可以指定開發環境設定要用哪個編譯器。

    2. 要修改CUDA for windows 的config header, host_condif.h 具體位置在: %HOME%\NVIDIA GPU Computing Toolkit\CUDA\v9.2\include\crt\host_config.h:133, 其中，HOME是你的設定;
    
       133 行是**#if _MSC_VER < 1600 || _MSC_VER > 1911** 看起來，是visaul studio版本年份問題（可能環境的年份給錯了？），之後我調整出來是：
         
     **#if _MSC_VER < 1600 || _MSC_VER > 1920** 
     
     頗累的。
     
    3. 然後windows平台上的utf8編碼都需要在檔案前加上BOM(這不設定話每次編譯都會得到你的檔案不是utf8編碼的回應), 能不能規避？不知道。
       
是說，去掉情緒字眼，內容少一半，突然想到，訓練個AI把文章情緒/疑似情緒字眼過濾掉，這樣是不是閱讀會更容易的(笑？
