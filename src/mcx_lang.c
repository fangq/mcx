/***************************************************************************//**
**  \mainpage Monte Carlo eXtreme - GPU accelerated Monte Carlo Photon Migration
**
**  \author Qianqian Fang <q.fang at neu.edu>
**  \copyright Qianqian Fang, 2009-2025
**
**  \section sref Reference
**  \li \c (\b Fang2009) Qianqian Fang and David A. Boas,
**          <a href="http://www.opticsinfobase.org/abstract.cfm?uri=oe-17-22-20178">
**          "Monte Carlo Simulation of Photon Migration in 3D Turbid Media Accelerated
**          by Graphics Processing Units,"</a> Optics Express, 17(22) 20178-20190 (2009).
**  \li \c (\b Yu2018) Leiming Yu, Fanny Nina-Paravecino, David Kaeli, and Qianqian Fang,
**          "Scalable and massively parallel Monte Carlo photon transport
**           simulations for heterogeneous computing platforms," J. Biomed. Optics,
**           23(1), 010504, 2018. https://doi.org/10.1117/1.JBO.23.1.010504
**  \li \c (\b Yan2020) Shijie Yan and Qianqian Fang* (2020), "Hybrid mesh and voxel
**          based Monte Carlo algorithm for accurate and efficient photon transport
**          modeling in complex bio-tissues," Biomed. Opt. Express, 11(11)
**          pp. 6262-6270. https://doi.org/10.1364/BOE.409468
**
**  \section sformat Formatting
**          Please always run "make pretty" inside the \c src folder before each commit.
**          The above command requires \c astyle to perform automatic formatting.
**
**  \section slicense License
**          GPL v3, see LICENSE.txt for details
*******************************************************************************/

/***************************************************************************//**
\file    mcx_lang.c

@brief   MCX language support

How to add a new translation:
First, decide the new language's ID - must be in the format of "aa_bb", where "aa"
must be the two-letter ISO 639 langauge code (https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes),
and "bb" must be the two-letter region code (https://en.wikipedia.org/wiki/List_of_ISO_3166_country_codes).
This code should be append to the below string array languagename[], and increase
the value of macro MAX_MCX_LANG defined in mcx_lang.h by 1.

Next, copy any one of the translation section below, starting from "MSTR(" and
ending with "})" and append to the end of the array translations[], right before
the last line "};" , add comma to separate this new translation JSON string with
its previous translations. The index of the translation strings must match
the location language ID in the languagename array.

Next, update the translation strings - the second string to the right of ":" on
each line to the desired language. DO NOT modify the English key names on the left,
because it could result in mismatch when performing the look-up. For "_MCX_BANNER_",
the translation is a multi-line string, please only modify the string content, and do not
remove or alter double-quotes/newlines or spaces after the content string in order
for this string to conform to JSON syntax while still be able to embed ASCII color
codes.

Once completed, plese compile MCX using make/make mex or cmake, and test your new
translation by adding "--lang/-y aa_BB" command line option, or setting cfg.lang in
mcxlab. Please also adjust the white-spaces in the printed _MCX_BANNER_ strings
to equalize the lengths of each line.
*******************************************************************************/

#include "mcx_lang.h"
#include "mcx_const.h"

#define MSTR(...) #__VA_ARGS__

const char *languagename[MAX_MCX_LANG] = {"zh_cn", "zh_tw", "ja_jp", "fr_ca", "es_mx", "de_de", "ko_kr", "hi_in", "ru_ru", "pt_br", ""};

const char* translations[MAX_MCX_LANG] = {
MSTR(
{
    "_LANG_":  "简体中文",
    "_LOCALE_":  "zh_CN",
    "_MCX_BANNER_": 
) "\"" S_MAGENTA "###############################################################################\n\
#                           极限蒙卡 (MCX) -- CUDA                            #\n\
#         作者版权 (c) 2009-2025 Qianqian Fang <q.fang at neu.edu>            #\n\
#" S_BLUE "                https://mcx.space/  &  https://neurojson.io                  " S_MAGENTA "#\n\
#                                                                             #\n\
# Computational Optics & Translational Imaging (COTI) Lab- " S_BLUE "http://fanglab.org " S_MAGENTA "#\n\
#               美国东北大学生物工程系，马萨诸塞州，波士顿                    #\n\
###############################################################################\n\
#   MCX 软件开发是在美国 NIH/NIGMS 经费(R01-GM114365)资助下完成的，特此致谢   #\n\
###############################################################################\n\
# 开源的科研代码以及开放的可重用科学数据对现代科学发展至关重要。MCX开发团队致 #\n\
# 力开放科学，并在 NIH 的资助下特别为此开发基于JSON的出入和输出文件格式。     #\n\
#                                                                             #\n\
# 请访问我们的开放数据门户网站 NeuroJSON.io(" S_BLUE "https://neurojson.io" S_MAGENTA ")，并诚挚邀请 #\n\
# 用户一道使用简单，可重用的 JSON 格式以及我们的免费网站来共享您的科研数据。  #\n" S_RESET "\"," MSTR(
    "absorbed":  "总吸收比例",
    "after encoding": "编码后压缩比",
    "A CUDA-capable GPU is not found or configured":  "找不到支持 CUDA 的 GPU",
    "Built-in benchmarks":  "内置仿真",
    "Built-in languages":  "内置语言",
    "code name":  "版本代号",
    "Command option":  "命令参数",
    "compiled by nvcc":  "nvcc 版本",
    "compiled with": "编译设置",
    "compressing data": "压缩数据",
    "compression ratio": "压缩比例",
    "data normalization complete":  "数据归一化完成",
    "detected": "探测到",
    "Downloading simulations from NeuroJSON.io (https://neurojson.org/db/mcx":  "从 NeuroJSON.io (https://neurojson.org/db/mcx) 下载仿真数据",
    "ERROR: No CUDA-capable GPU device found":  "错误：找不到支持 CUDA 的 GPU",
    "ERROR: Specified GPU ID is out of range":  "错误：指定 GPU 编号不在支持范围",
    "generating":  "生成",
    "GPU ID can not be more than 256":  "GPU 编号不可以超过256",
    "GPU ID must be non-zero":  "GPU 编号不可以为0",
    "GPU Information":  "GPU 信息",
    "GPU memory can not hold all time gates, disabling normalization to allow multiple runs":  "GPU 内存无法保存所有时间窗数据，关闭归一化设置",
    "incomplete input":  "输入不完整",
    "init complete":  "初始化完成",
    "initializing streams":  "初始化GPU",
    "input domain is 2D, the initial direction can not have non-zero value in the singular dimension":  "二维仿真中发车角度必须在二维平面中",
    "invalid json fragment following --json":  "--json输入格式错误",
    "Jacobian output is only valid in the reply mode. Please give an mch file after '-E'.":  "Jacobian 输出只能在重放模式使用。请在-E参数后提供 .mch 文件路径",
    "json fragment is expected after --json":  "--json参数后必须提供 JSON 格式数据",
    "json shape constructs are expected after -P":  "-P 参数后必须提供 JSON 格式的形状构件描述",
    "kernel complete":  "GPU 仿真程序完成",
    "launching MCX simulation for time window":  "运行 MCX 仿真，起止时间窗",
    "loss due to initial specular reflection is excluded in the total": "不包含初始入射时镜面反射的能量损失",
    "MCX Revision":  "MCX 版本",
    "MCX simulation speed":  "MCX 仿真速度",
    "No GPU device found":  "无法找到GPU",
    "normalization factor":  "归一化系数",
    "normalization factor for detector":  "探测器归一化系数",
    "normalizing raw data ...":  "归一化原始输出",
    "photons": "光子",
    "please use the -H option to specify a greater number":  "请使用 -H 参数指定更大的光子数",
    "please use the --maxjumpdebug option to specify a greater number":  "请使用 --maxjumpdebug 参数指定更大的光子数",
    "random numbers":  "随机数",
    "requesting shared memory":  "正在分配高速线程共享显存",
    "respin number can not be 0, check your -r/--repeat input or cfg.respin value":  "仿真分段数不可以为0; 请检查 -r/--repeat 输入参数或 cfg.respin 数值",
    "retrieving fields":  "获取仿真三维输出数组",
    "retrieving random numbers":  "获取生成的随机数",
    "saved trajectory positions":  "保存的",
    "saving data complete":  "保存数据完成",
    "saving data to file":  "保存数据到文件",
    "seed length": "随机数种子字(4字节)长",
    "seeding file is not supported in this binary":  "不支持保存随机数生成器种子",
    "simulated":  "模拟了",
    "simulation run#":  "仿真分段#",
    "source":  "光源",
    "the specified output data format is not recognized": "指定输出文件格式尚未支持",
    "the specified output data type is not recognized":  "指定输出数据类型尚未支持",
    "total simulated energy":  "仿真总能量",
    "transfer complete":  "获取数据完成",
    "unable to save to log file, will print from stdout":  "无法保存日志文件，只能从标准输出打印",
    "unknown short option":  "短参数不存在",
    "unknown verbose option":  "长参数不存在",
    "unnamed":  "未命名",
    "Unsupported bechmark":  "该仿真不存在",
    "Unsupported media format":  "指定媒质格式尚未支持",
    "WARNING: maxThreadsPerMultiProcessor can not be detected":  "警告：maxThreadsPerMultiProcessor 参数不支持",
    "WARNING: the detected photon number is more than what your have specified":  "警告：探测到的光子超过预先指定的数量",
    "WARNING: the saved trajectory positions are more than what your have specified":  "警告：保存的光子路径超过预先指定的数量",
    "WARNING: this MCX binary can not save partial path, please recompile mcx and make sure -D SAVE_DETECTORS is used by nvcc":  "警告：这个版本的 MCX 不支持保存 partial path，请添加 -D SAVE_DETECTORS 编译参数并重新编译 MCX",
    "workload was unspecified for an active device":  "启用的显卡无指定的仿真负载配比",
    "you can not specify both interactive mode and config file":  "配置文件(-f)和交互模式(-i)不可以同时使用"
}),

MSTR(
{
    "_LANG_": "繁體中文（台灣）",
    "_LOCALE_": "zh_TW",
    "_MCX_BANNER_": 
) "\"" S_MAGENTA "###############################################################################\n\
#                           極限蒙卡 (MCX) -- CUDA                            #\n\
#          作者版權 (c) 2009-2025 Qianqian Fang <q.fang at neu.edu>           #\n\
#" S_BLUE "                https://mcx.space/  &  https://neurojson.io                  " S_MAGENTA "#\n\
#                                                                             #\n\
#         計算光學與轉譯影像實驗室 (COTI Lab) - " S_BLUE "http://fanglab.org " S_MAGENTA "           #\n\
#               美國東北大學生物工程系，馬薩諸塞州波士頓                      #\n\
###############################################################################\n\
#   MCX 軟體開發獲得美國 NIH/NIGMS 經費(R01-GM114365)資助，在此致謝           #\n\
###############################################################################\n\
# 開源的科研代碼與可重複使用的開放科學數據對現代科學發展至關重要。MCX 開發團隊#\n\
# 致力於推動開放科學，並在NIH資助下，特別開發了基於JSON 的輸入與輸出檔案格式。#\n\
#                                                                             #\n\
# 請造訪我們的開放數據入口網站 NeuroJSON.io(" S_BLUE "https://neurojson.io" S_MAGENTA ")，誠摯邀請您 #\n\
# 一同使用簡單、可重複使用的 JSON 格式與我們的免費網站來分享您的科研數據。    #\n" S_RESET "\"," MSTR(
    "absorbed": "總吸收比例",
    "after encoding": "編碼後壓縮比",
    "A CUDA-capable GPU is not found or configured": "找不到支援 CUDA 的 GPU",
    "Built-in benchmarks": "內建模擬",
    "Built-in languages":  "内置语言",
    "code name": "版本代號",
    "Command option": "指令參數",
    "compiled by nvcc": "nvcc 編譯版本",
    "compiled with": "編譯設定",
    "compressing data": "壓縮資料",
    "compression ratio": "壓縮比例",
    "data normalization complete": "資料正規化完成",
    "detected": "偵測到",
    "Downloading simulations from NeuroJSON.io (https://neurojson.org/db/mcx": "從 NeuroJSON.io (https://neurojson.org/db/mcx) 下載模擬資料",
    "ERROR: No CUDA-capable GPU device found": "錯誤：找不到支援 CUDA 的 GPU 裝置",
    "ERROR: Specified GPU ID is out of range": "錯誤：指定的 GPU 編號超出範圍",
    "generating": "生成中",
    "GPU ID can not be more than 256": "GPU 編號不可超過 256",
    "GPU ID must be non-zero": "GPU 編號不可為 0",
    "GPU Information": "GPU 資訊",
    "GPU memory can not hold all time gates, disabling normalization to allow multiple runs": "GPU 記憶體不足以儲存所有時間窗，停用正規化以允許多次執行",
    "incomplete input": "輸入不完整",
    "init complete": "初始化完成",
    "initializing streams": "初始化 GPU",
    "input domain is 2D, the initial direction can not have non-zero value in the singular dimension": "在二維模擬中，初始方向不可有非零的單維分量",
    "invalid json fragment following --json": "--json 參數後的 JSON 格式錯誤",
    "Jacobian output is only valid in the reply mode. Please give an mch file after '-E'.": "Jacobian 輸出僅在回播模式中有效，請在 -E 參數後提供 .mch 檔案",
    "json fragment is expected after --json": "--json 參數後應提供 JSON 格式片段",
    "json shape constructs are expected after -P": "-P 參數後應提供 JSON 格式的形狀描述",
    "kernel complete": "GPU 模擬完成",
    "launching MCX simulation for time window": "啟動 MCX 模擬，時間窗：",
    "loss due to initial specular reflection is excluded in the total": "總能量不含初始鏡面反射損失",
    "MCX Revision": "MCX 版本",
    "MCX simulation speed": "MCX 模擬速度",
    "No GPU device found": "找不到 GPU 裝置",
    "normalization factor": "正規化係數",
    "normalization factor for detector": "偵測器正規化係數",
    "normalizing raw data ...": "正在正規化原始資料...",
    "photons": "光子",
    "please use the -H option to specify a greater number": "請使用 -H 參數指定更大的光子數",
    "please use the --maxjumpdebug option to specify a greater number": "請使用 --maxjumpdebug 參數指定更大的光子數",
    "random numbers": "隨機數",
    "requesting shared memory": "請求共用記憶體",
    "respin number can not be 0, check your -r/--repeat input or cfg.respin value": "模擬分段數不得為 0，請檢查 -r/--repeat 參數或 cfg.respin 設定值",
    "retrieving fields": "讀取模擬三維輸出欄位",
    "retrieving random numbers": "讀取隨機數",
    "saved trajectory positions": "儲存的軌跡位置",
    "saving data complete": "資料儲存完成",
    "saving data to file": "將資料儲存至檔案",
    "seed length": "隨機數種子長度（4位元組）",
    "seeding file is not supported in this binary": "此執行檔不支援種子檔功能",
    "simulated":  "模擬了",
    "simulation run#": "模擬執行次數 #",
    "source": "光源",
    "the specified output data format is not recognized": "不支援指定的輸出資料格式",
    "the specified output data type is not recognized": "不支援指定的輸出資料類型",
    "total simulated energy": "模擬總能量",
    "transfer complete": "資料傳輸完成",
    "unable to save to log file, will print from stdout": "無法寫入日誌檔案，將改為輸出至標準輸出",
    "unknown short option": "未知短參數",
    "unknown verbose option": "未知長參數",
    "unnamed": "未命名",
    "Unsupported bechmark": "不支援的模擬項目",
    "Unsupported media format": "不支援的媒質格式",
    "WARNING: maxThreadsPerMultiProcessor can not be detected": "警告：無法偵測 maxThreadsPerMultiProcessor",
    "WARNING: the detected photon number is more than what your have specified": "警告：偵測到的光子數超出指定數量",
    "WARNING: the saved trajectory positions are more than what your have specified": "警告：儲存的軌跡數超出指定數量",
    "WARNING: this MCX binary can not save partial path, please recompile mcx and make sure -D SAVE_DETECTORS is used by nvcc": "警告：此版本 MCX 不支援儲存 partial path，請使用 -D SAVE_DETECTORS 重新編譯",
    "workload was unspecified for an active device": "已啟用的裝置未設定工作負載比例",
    "you can not specify both interactive mode and config file": "不能同時指定互動模式(-i)與設定檔(-f)"
}),

MSTR(
{
    "_LANG_": "日本語",
    "_LOCALE_": "ja_JP",
    "_MCX_BANNER_": 
) "\"" S_MAGENTA "###############################################################################\n\
#                           極限モンテカルロ (MCX) -- CUDA                    #\n\
#         著作権 (c) 2009-2025 Qianqian Fang <q.fang at neu.edu>              #\n\
#" S_BLUE "                https://mcx.space/  &  https://neurojson.io                  " S_MAGENTA "#\n\
#                                                                             #\n\
# Computational Optics & Translational Imaging (COTI) Lab- " S_BLUE "http://fanglab.org " S_MAGENTA "#\n\
#   米国マサチューセッツ州ボストン、ノースイースタン大学 生体工学部           #\n\
###############################################################################\n\
#MCX ソフトウェアは米国 NIH/NIGMS の助成金(R01-GM114365)によって開発されました#\n\
###############################################################################\n\
# オープンソースの科学コードと再利用可能な科学データは、現代科学に不可欠です。#\n\
# MCX 開発チームはオープンサイエンスを推進し、NIH の支援を受け、JSON を用いた #\n\
# 入出力形式を特別に開発しました。                                            #\n\
#                                                                             #\n\
# 私たちのオープンデータポータルサイト NeuroJSON.io（" S_BLUE "https://neurojson.io" S_MAGENTA "）   #\n\
# を訪問し、簡単で再利用可能な JSON 形式と無料のウェブサービスを使って、あなた#\n\
# の研究データを#ぜひ共有してください。                                       #\n" S_RESET "\"," MSTR(
    "absorbed": "総吸収率",
    "after encoding": "エンコード後の圧縮率",
    "A CUDA-capable GPU is not found or configured": "CUDA対応のGPUが見つからないか、設定されていません",
    "Built-in benchmarks": "内蔵ベンチマーク",
    "Built-in languages":  "内蔵言語",
    "code name": "コードネーム",
    "Command option": "コマンドオプション",
    "compiled by nvcc": "nvcc でコンパイル",
    "compiled with": "コンパイル設定",
    "compressing data": "データを圧縮中",
    "compression ratio": "圧縮率",
    "data normalization complete": "データの正規化が完了しました",
    "detected": "検出された",
    "Downloading simulations from NeuroJSON.io (https://neurojson.org/db/mcx": "NeuroJSON.io (https://neurojson.org/db/mcx) からシミュレーションをダウンロード中",
    "ERROR: No CUDA-capable GPU device found": "エラー：CUDA対応GPUが見つかりません",
    "ERROR: Specified GPU ID is out of range": "エラー：指定されたGPU IDが範囲外です",
    "generating": "生成中",
    "GPU ID can not be more than 256": "GPU ID は 256 以下でなければなりません",
    "GPU ID must be non-zero": "GPU ID は 0 以外である必要があります",
    "GPU Information": "GPU 情報",
    "GPU memory can not hold all time gates, disabling normalization to allow multiple runs": "GPU メモリに全ての時間ゲートを保持できないため、正規化を無効にして複数回実行可能にします",
    "incomplete input": "入力が不完全です",
    "init complete": "初期化完了",
    "initializing streams": "GPUを初期化中",
    "input domain is 2D, the initial direction can not have non-zero value in the singular dimension": "2次元シミュレーションでは、単一次元の方向成分はゼロでなければなりません",
    "invalid json fragment following --json": "--json の後に無効な JSON フラグメントがあります",
    "Jacobian output is only valid in the reply mode. Please give an mch file after '-E'.": "Jacobian出力はリプレイモードでのみ有効です。'-E' の後に .mch ファイルを指定してください",
    "json fragment is expected after --json": "--json の後には JSON データが必要です",
    "json shape constructs are expected after -P": "-P の後には JSON 形式の形状データが必要です",
    "kernel complete": "GPU シミュレーション完了",
    "launching MCX simulation for time window": "指定した時間ウィンドウでMCXシミュレーションを開始中",
    "loss due to initial specular reflection is excluded in the total": "初期鏡面反射による損失は合計に含まれていません",
    "MCX Revision": "MCX バージョン",
    "MCX simulation speed": "MCX シミュレーション速度",
    "No GPU device found": "GPU デバイスが見つかりません",
    "normalization factor": "正規化係数",
    "normalization factor for detector": "検出器の正規化係数",
    "normalizing raw data ...": "生データを正規化中...",
    "photons": "光子",
    "please use the -H option to specify a greater number": "-H オプションを使ってより大きな値を指定してください",
    "please use the --maxjumpdebug option to specify a greater number": "--maxjumpdebug オプションでより大きな値を指定してください",
    "random numbers": "乱数",
    "requesting shared memory": "共有メモリをリクエスト中",
    "respin number can not be 0, check your -r/--repeat input or cfg.respin value": "再実行数は0にできません。-r/--repeatオプションかcfg.respinを確認してください",
    "retrieving fields": "3次元出力フィールドを取得中",
    "retrieving random numbers": "乱数を取得中",
    "saved trajectory positions": "保存された軌跡位置",
    "saving data complete": "データ保存完了",
    "saving data to file": "ファイルへデータを保存中",
    "seed length": "乱数種子の長さ（4バイト）",
    "seeding file is not supported in this binary": "このバイナリでは乱数種子ファイルはサポートされていません",
    "simulated":  "シミュレーションされた",
    "simulation run#": "シミュレーション実行 #",
    "source": "光源",
    "the specified output data format is not recognized": "指定された出力データ形式は未対応です",
    "the specified output data type is not recognized": "指定された出力データ型は未対応です",
    "total simulated energy": "総シミュレーションエネルギー",
    "transfer complete": "データ転送完了",
    "unable to save to log file, will print from stdout": "ログファイルに保存できないため、標準出力に表示します",
    "unknown short option": "不明な短オプション",
    "unknown verbose option": "不明な長オプション",
    "unnamed": "無名",
    "Unsupported bechmark": "このベンチマークはサポートされていません",
    "Unsupported media format": "指定されたメディア形式はサポートされていません",
    "WARNING: maxThreadsPerMultiProcessor can not be detected": "警告：maxThreadsPerMultiProcessor を検出できません",
    "WARNING: the detected photon number is more than what your have specified": "警告：検出された光子数が指定数を超えています",
    "WARNING: the saved trajectory positions are more than what your have specified": "警告：保存された軌跡数が指定数を超えています",
    "WARNING: this MCX binary can not save partial path, please recompile mcx and make sure -D SAVE_DETECTORS is used by nvcc": "警告：このMCXバイナリでは部分経路の保存ができません。-D SAVE_DETECTORSを使用して再コンパイルしてください",
    "workload was unspecified for an active device": "アクティブなデバイスに対して作業負荷が指定されていません",
    "you can not specify both interactive mode and config file": "インタラクティブモード（-i）と設定ファイル（-f）は同時に使用できません"
}),

MSTR(
{
    "_LANG_": "Français canadien",
    "_LOCALE_": "fr_CA",
    "_MCX_BANNER_": 
) "\"" S_MAGENTA "###############################################################################\n\
#                      Monte Carlo Extrême (MCX) -- CUDA                      #\n\
#     Droits d’auteur (c) 2009-2025 Qianqian Fang <q.fang at neu.edu>         #\n\
#" S_BLUE "                https://mcx.space/  &  https://neurojson.io                  " S_MAGENTA "#\n\
#                                                                             #\n\
#Optique computationnelle et imagerie translationnelle(COTI)-" S_BLUE "http://fanglab.org" S_MAGENTA "#\n\
# Département de bio-ingénierie, Université Northeastern, Boston, MA, É.-U.   #\n\
###############################################################################\n\
#Le développement de MCX est financé par le NIH/NIGMS des É.-U. (R01-GM114365)#\n\
###############################################################################\n\
# Le code scientifique libre et les données ouvertes réutilisables sont       #\n\
# essentiels à la science moderne. L’équipe MCX s’engage pour la science      #\n\
# ouverte et a développé un format d’entrée/sortie basé sur JSON grâce au     #\n\
# soutien du NIH.                                                             #\n\
#                                                                             #\n\
# Visitez notre portail de données ouvertes NeuroJSON.io(" S_BLUE "https://neurojson.io" S_MAGENTA ")#\n\
# et partagez vos données scientifiques à l’aide de JSON simple et            #\n\
# réutilisable, ainsi que notre site gratuit.                                 #\n" S_RESET "\"," MSTR(
    "absorbed": "taux d’absorption total",
    "after encoding": "Taux de compression après encodage",
    "A CUDA-capable GPU is not found or configured": "Aucun GPU compatible CUDA n’a été trouvé ou configuré",
    "Built-in benchmarks": "Tests de performance intégrés",
    "Built-in languages":  "Langues intégrées",
    "code name": "nom de code",
    "Command option": "option de commande",
    "compiled by nvcc": "compilé avec nvcc",
    "compiled with": "paramètres de compilation",
    "compressing data": "compression des données en cours",
    "compression ratio": "taux de compression",
    "data normalization complete": "normalisation des données terminée",
    "detected": "détecté",
    "Downloading simulations from NeuroJSON.io (https://neurojson.org/db/mcx": "Téléchargement des simulations depuis NeuroJSON.io (https://neurojson.org/db/mcx)",
    "ERROR: No CUDA-capable GPU device found": "ERREUR : Aucun périphérique GPU compatible CUDA trouvé",
    "ERROR: Specified GPU ID is out of range": "ERREUR : L’ID de GPU spécifié est hors plage",
    "generating": "génération en cours",
    "GPU ID can not be more than 256": "L’ID du GPU ne peut pas dépasser 256",
    "GPU ID must be non-zero": "L’ID du GPU ne peut pas être zéro",
    "GPU Information": "Informations sur le GPU",
    "GPU memory can not hold all time gates, disabling normalization to allow multiple runs": "La mémoire du GPU est insuffisante pour tous les intervalles temporels; normalisation désactivée pour permettre plusieurs exécutions",
    "incomplete input": "entrée incomplète",
    "init complete": "initialisation terminée",
    "initializing streams": "initialisation du GPU en cours",
    "input domain is 2D, the initial direction can not have non-zero value in the singular dimension": "Le domaine d’entrée est 2D; la direction initiale ne peut pas avoir de composante non nulle dans la dimension unique",
    "invalid json fragment following --json": "fragment JSON invalide après --json",
    "Jacobian output is only valid in the reply mode. Please give an mch file after '-E'.": "La sortie Jacobienne est seulement valide en mode de relecture. Veuillez fournir un fichier .mch après ‘-E’.",
    "json fragment is expected after --json": "un fragment JSON est attendu après --json",
    "json shape constructs are expected after -P": "des objets de forme JSON sont attendus après -P",
    "kernel complete": "exécution du noyau GPU terminée",
    "launching MCX simulation for time window": "lancement de la simulation MCX pour la fenêtre temporelle",
    "loss due to initial specular reflection is excluded in the total": "la perte due à la réflexion spéculaire initiale est exclue du total",
    "MCX Revision": "révision de MCX",
    "MCX simulation speed": "vitesse de simulation MCX",
    "No GPU device found": "aucun GPU détecté",
    "normalization factor": "facteur de normalisation",
    "normalization factor for detector": "facteur de normalisation du détecteur",
    "normalizing raw data ...": "normalisation des données brutes ...",
    "photons": "photons",
    "please use the -H option to specify a greater number": "veuillez utiliser l’option -H pour spécifier un nombre plus élevé",
    "please use the --maxjumpdebug option to specify a greater number": "veuillez utiliser l’option --maxjumpdebug pour spécifier un nombre plus élevé",
    "random numbers": "nombres aléatoires",
    "requesting shared memory": "allocation de mémoire partagée en cours",
    "respin number can not be 0, check your -r/--repeat input or cfg.respin value": "le nombre de répétitions ne peut pas être 0; vérifiez l’entrée -r/--repeat ou cfg.respin",
    "retrieving fields": "récupération des champs de sortie 3D",
    "retrieving random numbers": "récupération des nombres aléatoires",
    "saved trajectory positions": "positions de trajectoires enregistrées",
    "saving data complete": "sauvegarde des données terminée",
    "saving data to file": "sauvegarde des données dans un fichier",
    "seed length": "longueur de la graine (4 octets)",
    "seeding file is not supported in this binary": "le fichier de graine aléatoire n’est pas pris en charge dans ce binaire",
    "simulated":  "simulé",
    "simulation run#": "exécution de simulation #",
    "source": "source",
    "the specified output data format is not recognized": "le format de sortie spécifié n’est pas pris en charge",
    "the specified output data type is not recognized": "le type de données de sortie spécifié n’est pas reconnu",
    "total simulated energy": "énergie totale simulée",
    "transfer complete": "transfert terminé",
    "unable to save to log file, will print from stdout": "impossible d’écrire dans le fichier journal; impression via la sortie standard",
    "unknown short option": "option abrégée inconnue",
    "unknown verbose option": "option complète inconnue",
    "unnamed": "sans nom",
    "Unsupported bechmark": "test de performance non pris en charge",
    "Unsupported media format": "format de média non pris en charge",
    "WARNING: maxThreadsPerMultiProcessor can not be detected": "AVERTISSEMENT : maxThreadsPerMultiProcessor n’a pas pu être détecté",
    "WARNING: the detected photon number is more than what your have specified": "AVERTISSEMENT : le nombre de photons détectés dépasse le nombre spécifié",
    "WARNING: the saved trajectory positions are more than what your have specified": "AVERTISSEMENT : le nombre de trajectoires enregistrées dépasse celui spécifié",
    "WARNING: this MCX binary can not save partial path, please recompile mcx and make sure -D SAVE_DETECTORS is used by nvcc": "AVERTISSEMENT : ce binaire de MCX ne peut pas enregistrer de chemin partiel; veuillez recompiler avec -D SAVE_DETECTORS",
    "workload was unspecified for an active device": "aucune charge de travail spécifiée pour un appareil actif",
    "you can not specify both interactive mode and config file": "vous ne pouvez pas spécifier à la fois le mode interactif (-i) et un fichier de configuration (-f)"
}),

MSTR(
{
    "_LANG_": "Español mexicano",
    "_LOCALE_": "es_MX",
    "_MCX_BANNER_": 
) "\"" S_MAGENTA "###############################################################################\n\
#                  Monte Carlo Extremo (MCX) -- CUDA                          #\n\
#     Derechos de autor (c) 2009-2025 Qianqian Fang <q.fang at neu.edu>       #\n\
#" S_BLUE "                https://mcx.space/  &  https://neurojson.io                  " S_MAGENTA "#\n\
#                                                                             #\n\
# Laboratorio de Óptica Computacional e Imágenes Traslacionales (COTI) - " S_BLUE "http://fanglab.org " S_MAGENTA "#\n\
# Departamento de Bioingeniería, Universidad de Northeastern, Boston, MA, EE. UU. #\n\
###############################################################################\n\
#   El desarrollo del software MCX fue financiado por el NIH/NIGMS de EE. UU. (R01-GM114365) #\n\
###############################################################################\n\
# El código científico de código abierto y los datos científicos reutilizables son esenciales para el avance de la ciencia moderna. El equipo de desarrollo de MCX está comprometido con la ciencia abierta y, con el apoyo del NIH, ha desarrollado un formato de entrada y salida basado en JSON. #\n\
#                                                                             #\n\
# Visita nuestro portal de datos abiertos NeuroJSON.io (" S_BLUE "https://neurojson.io" S_MAGENTA ") e invita a otros a compartir tus datos científicos utilizando el formato JSON simple y reutilizable, así como nuestro sitio web gratuito. #\n" S_RESET "\"," MSTR(
    "absorbed": "proporción total absorbida",
    "after encoding": "Tasa de compresión después de la codificación",
    "A CUDA-capable GPU is not found or configured": "No se encontró ni se configuró una GPU compatible con CUDA",
    "Built-in benchmarks": "Pruebas integradas",
    "Built-in languages":  "Idiomas integrados",
    "code name": "nombre de código",
    "Command option": "opción de comando",
    "compiled by nvcc": "compilado por nvcc",
    "compiled with": "compilado con",
    "compressing data": "comprimiendo datos",
    "compression ratio": "tasa de compresión",
    "data normalization complete": "normalización de datos completada",
    "detected": "detectado",
    "Downloading simulations from NeuroJSON.io (https://neurojson.org/db/mcx": "Descargando simulaciones desde NeuroJSON.io (https://neurojson.org/db/mcx)",
    "ERROR: No CUDA-capable GPU device found": "ERROR: No se encontró un dispositivo GPU compatible con CUDA",
    "ERROR: Specified GPU ID is out of range": "ERROR: El ID de GPU especificado está fuera de rango",
    "generating": "generando",
    "GPU ID can not be more than 256": "El ID de GPU no puede ser mayor que 256",
    "GPU ID must be non-zero": "El ID de GPU debe ser diferente de cero",
    "GPU Information": "Información de la GPU",
    "GPU memory can not hold all time gates, disabling normalization to allow multiple runs": "La memoria de la GPU no puede contener todas las ventanas de tiempo, desactivando la normalización para permitir múltiples ejecuciones",
    "incomplete input": "entrada incompleta",
    "init complete": "inicialización completada",
    "initializing streams": "inicializando flujos",
    "input domain is 2D, the initial direction can not have non-zero value in the singular dimension": "el dominio de entrada es 2D, la dirección inicial no puede tener un valor distinto de cero en la dimensión singular",
    "invalid json fragment following --json": "fragmento JSON inválido después de --json",
    "Jacobian output is only valid in the reply mode. Please give an mch file after '-E'.": "La salida Jacobiana solo es válida en el modo de respuesta. Por favor, proporciona un archivo .mch después de '-E'.",
    "json fragment is expected after --json": "se espera un fragmento JSON después de --json",
    "json shape constructs are expected after -P": "se esperan construcciones de forma JSON después de -P",
    "kernel complete": "núcleo completado",
    "launching MCX simulation for time window": "iniciando simulación MCX para la ventana de tiempo",
    "loss due to initial specular reflection is excluded in the total": "la pérdida debido a la reflexión especular inicial está excluida del total",
    "MCX Revision": "Revisión de MCX",
    "MCX simulation speed": "velocidad de simulación MCX",
    "No GPU device found": "No se encontró ningún dispositivo GPU",
    "normalization factor": "factor de normalización",
    "normalization factor for detector": "factor de normalización para el detector",
    "normalizing raw data ...": "normalizando datos sin procesar ...",
    "photons": "fotones",
    "please use the -H option to specify a greater number": "por favor, usa la opción -H para especificar un número mayor",
    "please use the --maxjumpdebug option to specify a greater number": "por favor, usa la opción --maxjumpdebug para especificar un número mayor",
    "random numbers": "números aleatorios",
    "requesting shared memory": "solicitando memoria compartida",
    "respin number can not be 0, check your -r/--repeat input or cfg.respin value": "el número de repeticiones no puede ser 0, verifica tu entrada -r/--repeat o el valor de cfg.respin",
    "retrieving fields": "recuperando campos",
    "retrieving random numbers": "recuperando números aleatorios",
    "saved trajectory positions": "posiciones de trayectoria guardadas",
    "saving data complete": "guardado de datos completado",
    "saving data to file": "guardando datos en archivo",
    "seed length": "longitud de la semilla",
    "seeding file is not supported in this binary": "el archivo de semilla no es compatible en este binario",
    "simulated":  "simulado",
    "simulation run#": "ejecución de simulación #",
    "source": "fuente",
    "the specified output data format is not recognized": "el formato de datos de salida especificado no es reconocido",
    "the specified output data type is not recognized": "el tipo de datos de salida especificado no es reconocido",
    "total simulated energy": "energía total simulada",
    "transfer complete": "transferencia completada",
    "unable to save to log file, will print from stdout": "no se puede guardar en el archivo de registro, se imprimirá desde stdout",
    "unknown short option": "opción corta desconocida",
    "unknown verbose option": "opción detallada desconocida",
    "unnamed": "sin nombre",
    "Unsupported bechmark": "prueba no compatible",
    "Unsupported media format": "formato de medio no compatible",
    "WARNING: maxThreadsPerMultiProcessor can not be detected": "ADVERTENCIA: maxThreadsPerMultiProcessor no se puede detectar",
    "WARNING: the detected photon number is more than what your have specified": "ADVERTENCIA: el número de fotones detectado es mayor que el especificado",
    "WARNING: the saved trajectory positions are more than what your have specified": "ADVERTENCIA: las posiciones de trayectoria guardadas son más que las especificadas",
    "WARNING: this MCX binary can not save partial path, please recompile mcx and make sure -D SAVE_DETECTORS is used by nvcc": "ADVERTENCIA: este binario de MCX no puede guardar rutas parciales, por favor recompila mcx y asegúrate de que -D SAVE_DETECTORS sea usado por nvcc",
    "workload was unspecified for an active device": "la carga de trabajo no fue especificada para un dispositivo activo",
    "you can not specify both interactive mode and config file": "no puedes especificar tanto el modo interactivo como el archivo de configuración"
}),

MSTR(
{
    "_LANG_": "Deutsch",
    "_LOCALE_": "de_DE",
    "_MCX_BANNER_": 
) "\"" S_MAGENTA "###############################################################################\n\
#                         MCX – Monte Carlo eXtreme -- CUDA                   #\n\
#       Urheberrecht (c) 2009–2025 Qianqian Fang <q.fang at neu.edu>          #\n\
#" S_BLUE "                https://mcx.space/  &  https://neurojson.io                  " S_MAGENTA "#\n\
#                                                                             #\n\
# Computational Optics & Translational Imaging (COTI) Lab – " S_BLUE "http://fanglab.org" S_MAGENTA "#\n\
#   Department für Bioengineering, Northeastern University, Boston, MA, USA   #\n\
###############################################################################\n\
#   Die Entwicklung der MCX-Software wurde vom US-amerikanischen NIH/NIGMS    #\n\
#   (R01-GM114365) gefördert – vielen Dank für die Unterstützung.             #\n\
###############################################################################\n\
# Offener wissenschaftlicher Code und wiederverwendbare wissenschaftliche     #\n\
# Daten sind entscheidend für den Fortschritt moderner Wissenschaft. Das MCX- #\n\
# Team fördert Open Science aktiv und hat mit Unterstützung des NIH ein JSON- #\n\
# basiertes Datenformat für Ein- und Ausgaben entwickelt.                     #\n\
#                                                                             #\n\
# Besuchen Sie unser offenes Datenportal NeuroJSON.io (" S_BLUE "https://neurojson.io" S_MAGENTA ")  #\n\
# und teilen Sie Ihre wissenschaftlichen Daten im wiederverwendbaren JSON-    #\n\
# Format über unsere kostenlose Webseite.                                     #\n" S_RESET "\"," MSTR(
    "absorbed": "Gesamte Absorptionsrate",
    "after encoding": "Komprimierungsrate nach der Kodierung",
    "A CUDA-capable GPU is not found or configured": "Keine CUDA-fähige GPU gefunden oder konfiguriert",
    "Built-in benchmarks": "Integrierte Benchmarks",
    "Built-in languages":  "Integrierte Sprachen",
    "code name": "Codename",
    "Command option": "Kommandooption",
    "compiled by nvcc": "Kompiliert mit nvcc",
    "compiled with": "Kompiliert mit",
    "compressing data": "Daten werden komprimiert",
    "compression ratio": "Kompressionsverhältnis",
    "data normalization complete": "Daten-Normalisierung abgeschlossen",
    "detected": "Erkannt",
    "Downloading simulations from NeuroJSON.io (https://neurojson.org/db/mcx": "Simulationen werden von NeuroJSON.io heruntergeladen (https://neurojson.org/db/mcx)",
    "ERROR: No CUDA-capable GPU device found": "FEHLER: Keine CUDA-fähige GPU gefunden",
    "ERROR: Specified GPU ID is out of range": "FEHLER: Angegebene GPU-ID liegt außerhalb des gültigen Bereichs",
    "generating": "wird generiert",
    "GPU ID can not be more than 256": "GPU-ID darf 256 nicht überschreiten",
    "GPU ID must be non-zero": "GPU-ID darf nicht null sein",
    "GPU Information": "GPU-Informationen",
    "GPU memory can not hold all time gates, disabling normalization to allow multiple runs": "GPU-Speicher reicht nicht für alle Zeitfenster, Normalisierung wird deaktiviert für mehrere Durchläufe",
    "incomplete input": "Unvollständige Eingabe",
    "init complete": "Initialisierung abgeschlossen",
    "initializing streams": "Initialisiere Streams",
    "input domain is 2D, the initial direction can not have non-zero value in the singular dimension": "Bei 2D-Simulationen darf die Anfangsrichtung in der dritten Dimension keinen Wert ungleich null haben",
    "invalid json fragment following --json": "Ungültiges JSON-Fragment nach --json",
    "Jacobian output is only valid in the reply mode. Please give an mch file after '-E'.": "Jacobi-Ausgabe ist nur im Replay-Modus gültig. Bitte geben Sie eine .mch-Datei nach '-E' an.",
    "json fragment is expected after --json": "JSON-Fragment wird nach --json erwartet",
    "json shape constructs are expected after -P": "JSON-Formbeschreibung wird nach -P erwartet",
    "kernel complete": "Kernel-Ausführung abgeschlossen",
    "launching MCX simulation for time window": "Starte MCX-Simulation für Zeitfenster",
    "loss due to initial specular reflection is excluded in the total": "Verlust durch anfängliche Spiegelreflexion nicht in der Gesamtenergie enthalten",
    "MCX Revision": "MCX-Version",
    "MCX simulation speed": "MCX-Simulationsgeschwindigkeit",
    "No GPU device found": "Kein GPU-Gerät gefunden",
    "normalization factor": "Normalisierungsfaktor",
    "normalization factor for detector": "Normalisierungsfaktor für Detektor",
    "normalizing raw data ...": "Rohdaten werden normalisiert ...",
    "photons": "photonen",
    "please use the -H option to specify a greater number": "Bitte verwenden Sie die Option -H, um eine höhere Zahl anzugeben",
    "please use the --maxjumpdebug option to specify a greater number": "Bitte verwenden Sie --maxjumpdebug, um eine höhere Zahl anzugeben",
    "random numbers": "Zufallszahlen",
    "requesting shared memory": "Fordere Shared Memory an",
    "respin number can not be 0, check your -r/--repeat input or cfg.respin value": "Wiederholungsanzahl darf nicht 0 sein; prüfen Sie -r/--repeat oder cfg.respin",
    "retrieving fields": "Felder werden abgerufen",
    "retrieving random numbers": "Zufallszahlen werden abgerufen",
    "saved trajectory positions": "Gespeicherte Trajektorienpositionen",
    "saving data complete": "Daten erfolgreich gespeichert",
    "saving data to file": "Speichere Daten in Datei",
    "seed length": "Länge des Seeds (4 Bytes)",
    "seeding file is not supported in this binary": "Seed-Datei wird in dieser Version nicht unterstützt",
    "simulated":  "simuliert",
    "simulation run#": "Simulationslauf #",
    "source": "Quelle",
    "the specified output data format is not recognized": "Das angegebene Ausgabedatenformat wird nicht erkannt",
    "the specified output data type is not recognized": "Der angegebene Datentyp wird nicht erkannt",
    "total simulated energy": "Gesamtsimulierte Energie",
    "transfer complete": "Übertragung abgeschlossen",
    "unable to save to log file, will print from stdout": "Protokoll kann nicht gespeichert werden, Ausgabe erfolgt über stdout",
    "unknown short option": "Unbekannte Kurzoption",
    "unknown verbose option": "Unbekannte Langoption",
    "unnamed": "Unbenannt",
    "Unsupported bechmark": "Nicht unterstützter Benchmark",
    "Unsupported media format": "Nicht unterstütztes Medienformat",
    "WARNING: maxThreadsPerMultiProcessor can not be detected": "WARNUNG: maxThreadsPerMultiProcessor konnte nicht erkannt werden",
    "WARNING: the detected photon number is more than what your have specified": "WARNUNG: Detektierte Photonenzahl überschreitet den angegebenen Wert",
    "WARNING: the saved trajectory positions are more than what your have specified": "WARNUNG: Gespeicherte Trajektorien überschreiten den angegebenen Wert",
    "WARNING: this MCX binary can not save partial path, please recompile mcx and make sure -D SAVE_DETECTORS is used by nvcc": "WARNUNG: Diese MCX-Version kann keine Teilpfade speichern. Bitte kompilieren Sie MCX neu mit der Option -D SAVE_DETECTORS",
    "workload was unspecified for an active device": "Keine Arbeitslast für aktives Gerät angegeben",
    "you can not specify both interactive mode and config file": "Interaktiver Modus (-i) und Konfigurationsdatei (-f) dürfen nicht gleichzeitig verwendet werden"
}),

MSTR(
{
    "_LANG_": "한국어",
    "_LOCALE_": "ko_KR",
    "_MCX_BANNER_": 
) "\"" S_MAGENTA "###############################################################################\n\
#                           MCX – 몬테카를로 익스트림 (CUDA)                  #\n\
#         저작권 (c) 2009–2025 Qianqian Fang <q.fang at neu.edu>              #\n\
#" S_BLUE "                https://mcx.space/  &  https://neurojson.io                  " S_MAGENTA "#\n\
#                                                                             #\n\
#    계산 광학 및 변환 영상 연구실(COTI Lab) – " S_BLUE "http://fanglab.org             " S_MAGENTA "#\n\
#   미국 매사추세츠주 보스턴, 노스이스턴 대학교 생명공학과                    #\n\
###############################################################################\n\
#MCX 소프트웨어 개발은 미국 NIH/NIGMS의 지원(R01-GM114365)으로 이루어졌습니다.#\n\
###############################################################################\n\
# 오픈 소스 연구 코드와 재사용 가능한 과학 데이터는 현대 과학 발전에 필수적입니다. #\n\
# MCX 개발팀은 NIH의 지원을 받아 JSON 기반의 입출력 파일 형식을 개발하였습니다.  #\n\
#                                                                             #\n\
# 우리의 오픈 데이터 포털인 NeuroJSON.io(" S_BLUE "https://neurojson.io" S_MAGENTA ")를 방문하시고,  #\n\
# 간단하고 재사용 가능한JSON형식과 무료 웹사이트를 통해연구데이터를 공유해 주세요. #\n" S_RESET "\"," MSTR(
    "absorbed": "총 흡수 비율",
    "after encoding": "인코딩 후 압축 비율",
    "A CUDA-capable GPU is not found or configured": "CUDA를 지원하는 GPU를 찾을 수 없거나 구성되지 않았습니다",
    "Built-in benchmarks": "내장 벤치마크",
    "Built-in languages":  "내장 언어",
    "code name": "코드 이름",
    "Command option": "명령 옵션",
    "compiled by nvcc": "nvcc로 컴파일됨",
    "compiled with": "다음으로 컴파일됨",
    "compressing data": "데이터 압축 중",
    "compression ratio": "압축 비율",
    "data normalization complete": "데이터 정규화 완료",
    "detected": "감지됨",
    "Downloading simulations from NeuroJSON.io (https://neurojson.org/db/mcx": "NeuroJSON.io에서 시뮬레이션 다운로드 중 (https://neurojson.org/db/mcx)",
    "ERROR: No CUDA-capable GPU device found": "오류: CUDA를 지원하는 GPU 장치를 찾을 수 없습니다",
    "ERROR: Specified GPU ID is out of range": "오류: 지정된 GPU ID가 범위를 벗어났습니다",
    "generating": "생성 중",
    "GPU ID can not be more than 256": "GPU ID는 256을 초과할 수 없습니다",
    "GPU ID must be non-zero": "GPU ID는 0이 될 수 없습니다",
    "GPU Information": "GPU 정보",
    "GPU memory can not hold all time gates, disabling normalization to allow multiple runs": "GPU 메모리에 모든 시간 게이트를 저장할 수 없어 정규화를 비활성화하고 여러 실행을 허용합니다",
    "incomplete input": "입력이 불완전합니다",
    "init complete": "초기화 완료",
    "initializing streams": "스트림 초기화 중",
    "input domain is 2D, the initial direction can not have non-zero value in the singular dimension": "입력 도메인이 2D이므로 초기 방향은 단일 차원에서 0이 아닌 값을 가질 수 없습니다",
    "invalid json fragment following --json": "--json 뒤에 잘못된 JSON 조각이 있습니다",
    "Jacobian output is only valid in the reply mode. Please give an mch file after '-E'.": "Jacobian 출력은 재생 모드에서만 유효합니다. '-E' 뒤에 mch 파일을 제공해 주세요.",
    "json fragment is expected after --json": "--json 뒤에 JSON 조각이 필요합니다",
    "json shape constructs are expected after -P": "-P 뒤에 JSON 형태 구성이 필요합니다",
    "kernel complete": "커널 완료",
    "launching MCX simulation for time window": "시간 창에 대한 MCX 시뮬레이션 시작",
    "loss due to initial specular reflection is excluded in the total": "초기 거울 반사로 인한 손실은 총합에서 제외됩니다",
    "MCX Revision": "MCX 개정판",
    "MCX simulation speed": "MCX 시뮬레이션 속도",
    "No GPU device found": "GPU 장치를 찾을 수 없습니다",
    "normalization factor": "정규화 계수",
    "normalization factor for detector": "검출기 정규화 계수",
    "normalizing raw data ...": "원시 데이터 정규화 중 ...",
    "photons": "광자들",
    "please use the -H option to specify a greater number": "더 큰 수를 지정하려면 -H 옵션을 사용하세요",
    "please use the --maxjumpdebug option to specify a greater number": "더 큰 수를 지정하려면 --maxjumpdebug 옵션을 사용하세요",
    "random numbers": "난수",
    "requesting shared memory": "공유 메모리 요청 중",
    "respin number can not be 0, check your -r/--repeat input or cfg.respin value": "respin 수는 0이 될 수 없습니다. -r/--repeat 입력 또는 cfg.respin 값을 확인하세요",
    "retrieving fields": "필드 검색 중",
    "retrieving random numbers": "난수 검색 중",
    "saved trajectory positions": "저장된 궤적 위치",
    "saving data complete": "데이터 저장 완료",
    "saving data to file": "파일에 데이터 저장 중",
    "seed length": "시드 길이",
    "seeding file is not supported in this binary": "이 바이너리에서는 시드 파일이 지원되지 않습니다",
    "simulated":  "시뮬레이션됨",
    "simulation run#": "시뮬레이션 실행#",
    "source": "소스",
    "the specified output data format is not recognized": "지정된 출력 데이터 형식을 인식할 수 없습니다",
    "the specified output data type is not recognized": "지정된 출력 데이터 유형을 인식할 수 없습니다",
    "total simulated energy": "총 시뮬레이션 에너지",
    "transfer complete": "전송 완료",
    "unable to save to log file, will print from stdout": "로그 파일에 저장할 수 없어 stdout에 출력합니다",
    "unknown short option": "알 수 없는 짧은 옵션",
    "unknown verbose option": "알 수 없는 자세한 옵션",
    "unnamed": "이름 없음",
    "Unsupported bechmark": "지원되지 않는 벤치마크",
    "Unsupported media format": "지원되지 않는 미디어 형식",
    "WARNING: maxThreadsPerMultiProcessor can not be detected": "경고: maxThreadsPerMultiProcessor를 감지할 수 없습니다",
    "WARNING: the detected photon number is more than what your have specified": "경고: 감지된 광자 수가 지정한 수를 초과합니다",
    "WARNING: the saved trajectory positions are more than what your have specified": "경고: 저장된 궤적 위치 수가 지정한 수를 초과합니다",
    "WARNING: this MCX binary can not save partial path, please recompile mcx and make sure -D SAVE_DETECTORS is used by nvcc": "경고: 이 MCX 바이너리는 부분 경로를 저장할 수 없습니다. mcx를 다시 컴파일하고 nvcc에 -D SAVE_DETECTORS를 사용했는지 확인하세요",
    "workload was unspecified for an active device": "활성 장치에 대한 작업 부하가 지정되지 않았습니다",
    "you can not specify both interactive mode and config file": "대화형 모드와 구성 파일을 동시에 지정할 수 없습니다"
}),

MSTR(
{
    "_LANG_": "हिन्दी",
    "_LOCALE_": "hi_IN",
    "_MCX_BANNER_": 
) "\"" S_MAGENTA "###############################################################################\n\
#                            MCX – मोंटे कार्लो एक्सट्रीम -- CUDA                  #\n\
#       कॉपीराइट (c) 2009–2025 कियानकियान फांग <q.fang at neu.edu>             #\n\
#" S_BLUE "                https://mcx.space/  &  https://neurojson.io                  " S_MAGENTA "#\n\
#                                                                             #\n\
#         कम्प्यूटेशनल ऑप्टिक्स और ट्रांसलेशनल इमेजिंग (COTI) लैब – " S_BLUE "http://fanglab.org    " S_MAGENTA "#\n\
#          बायोइंजीनियरिंग विभाग, नॉर्थईस्टर्न यूनिवर्सिटी, बोस्टन, MA, USA           #\n\
###############################################################################\n\
#   MCX सॉफ़्टवेयर का विकास अमेरिकी NIH/NIGMS (R01-GM114365) द्वारा समर्थित है –     #\n\
#   समर्थन के लिए धन्यवाद।                                                       #\n\
###############################################################################\n\
# ओपन-सोर्स वैज्ञानिक कोड और पुन: प्रयोज्य वैज्ञानिक डेटा आधुनिक विज्ञान की प्रगति के लिए आवश्यक हैं।     #\n\
# MCX टीम ओपन साइंस को सक्रिय रूप से बढ़ावा देती है और NIH के समर्थन से एक JSON-आधारित इनपुट/आउटपुट डेटा प्रारूप विकसित किया है। #\n\
#                                                                             #\n\
# हमारे ओपन डेटा पोर्टल NeuroJSON.io (" S_BLUE "https://neurojson.io" S_MAGENTA ") पर जाएं और हमारे मुफ्त वेबसाइट के माध्यम से #\n\
# पुन: प्रयोज्य JSON प्रारूप में अपने वैज्ञानिक डेटा साझा करें।              #\n" S_RESET "\"," MSTR(
    "absorbed": "कुल अवशोषण दर",
    "after encoding": "एन्कोडिंग के बाद संपीड़न अनुपात",
    "A CUDA-capable GPU is not found or configured": "CUDA-सक्षम GPU नहीं मिला या कॉन्फ़िगर नहीं किया गया है",
    "Built-in benchmarks": "इनबिल्ट बेंचमार्क",
    "Built-in languages":  "अंतर्निर्मित भाषाएँ",
    "code name": "कोड नाम",
    "Command option": "कमांड विकल्प",
    "compiled by nvcc": "nvcc द्वारा संकलित",
    "compiled with": "के साथ संकलित",
    "compressing data": "डेटा संपीड़न हो रहा है",
    "compression ratio": "संपीड़न अनुपात",
    "data normalization complete": "डेटा सामान्यीकरण पूर्ण",
    "detected": "पता चला",
    "Downloading simulations from NeuroJSON.io (https://neurojson.org/db/mcx": "NeuroJSON.io से सिमुलेशन डाउनलोड हो रहा है (https://neurojson.org/db/mcx)",
    "ERROR: No CUDA-capable GPU device found": "त्रुटि: कोई CUDA-सक्षम GPU डिवाइस नहीं मिला",
    "ERROR: Specified GPU ID is out of range": "त्रुटि: निर्दिष्ट GPU ID सीमा से बाहर है",
    "generating": "उत्पन्न किया जा रहा है",
    "GPU ID can not be more than 256": "GPU ID 256 से अधिक नहीं हो सकता",
    "GPU ID must be non-zero": "GPU ID शून्य नहीं हो सकता",
    "GPU Information": "GPU जानकारी",
    "GPU memory can not hold all time gates, disabling normalization to allow multiple runs": "GPU मेमोरी सभी समय गेट्स को समाहित नहीं कर सकती, कई रन की अनुमति देने के लिए सामान्यीकरण अक्षम किया जा रहा है",
    "incomplete input": "अपूर्ण इनपुट",
    "init complete": "प्रारंभ पूर्ण",
    "initializing streams": "स्ट्रीम्स प्रारंभ हो रही हैं",
    "input domain is 2D, the initial direction can not have non-zero value in the singular dimension": "इनपुट डोमेन 2D है, प्रारंभिक दिशा में एकल आयाम में शून्य से भिन्न मान नहीं हो सकता",
    "invalid json fragment following --json": "--json के बाद अमान्य JSON खंड",
    "Jacobian output is only valid in the reply mode. Please give an mch file after '-E'.": "जैकॉबियन आउटपुट केवल रिप्लाई मोड में मान्य है। कृपया '-E' के बाद एक mch फ़ाइल दें।",
    "json fragment is expected after --json": "--json के बाद JSON खंड अपेक्षित है",
    "json shape constructs are expected after -P": "-P के बाद JSON आकार संरचनाएं अपेक्षित हैं",
    "kernel complete": "कर्नेल पूर्ण",
    "launching MCX simulation for time window": "समय विंडो के लिए MCX सिमुलेशन प्रारंभ हो रहा है",
    "loss due to initial specular reflection is excluded in the total": "प्रारंभिक परावर्तक परावर्तन के कारण होने वाला नुकसान कुल में शामिल नहीं है",
    "MCX Revision": "MCX संस्करण",
    "MCX simulation speed": "MCX सिमुलेशन गति",
    "No GPU device found": "कोई GPU डिवाइस नहीं मिला",
    "normalization factor": "सामान्यीकरण गुणांक",
    "normalization factor for detector": "डिटेक्टर के लिए सामान्यीकरण गुणांक",
    "normalizing raw data ...": "कच्चे डेटा का सामान्यीकरण हो रहा है ...",
    "photons": "फोटॉन्स",
    "please use the -H option to specify a greater number": "कृपया अधिक संख्या निर्दिष्ट करने के लिए -H विकल्प का उपयोग करें",
    "please use the --maxjumpdebug option to specify a greater number": "कृपया अधिक संख्या निर्दिष्ट करने के लिए --maxjumpdebug विकल्प का उपयोग करें",
    "random numbers": "यादृच्छिक संख्याएँ",
    "requesting shared memory": "साझा मेमोरी का अनुरोध किया जा रहा है",
    "respin number can not be 0, check your -r/--repeat input or cfg.respin value": "respin संख्या 0 नहीं हो सकती, कृपया अपने -r/--repeat इनपुट या cfg.respin मान की जांच करें",
    "retrieving fields": "फ़ील्ड्स प्राप्त किए जा रहे हैं",
    "retrieving random numbers": "यादृच्छिक संख्याएँ प्राप्त की जा रही हैं",
    "saved trajectory positions": "सहेजे गए प्रक्षेपवक्र स्थितियाँ",
    "saving data complete": "डेटा सहेजना पूर्ण",
    "saving data to file": "डेटा फ़ाइल में सहेजा जा रहा है",
    "seed length": "सीड लंबाई",
    "seeding file is not supported in this binary": "इस बाइनरी में सीड फ़ाइल समर्थित नहीं है",
    "simulated":  "अनुकरण किया गया",
    "simulation run#": "सिमुलेशन रन#",
    "source": "स्रोत",
    "the specified output data format is not recognized": "निर्दिष्ट आउटपुट डेटा प्रारूप मान्यता प्राप्त नहीं है",
    "the specified output data type is not recognized": "निर्दिष्ट आउटपुट डेटा प्रकार मान्यता प्राप्त नहीं है",
    "total simulated energy": "कुल सिमुलेटेड ऊर्जा",
    "transfer complete": "स्थानांतरण पूर्ण",
    "unable to save to log file, will print from stdout": "लॉग फ़ाइल में सहेजने में असमर्थ, stdout से प्रिंट किया जाएगा",
    "unknown short option": "अज्ञात शॉर्ट विकल्प",
    "unknown verbose option": "अज्ञात विस्तृत विकल्प",
    "unnamed": "बेनाम",
    "Unsupported bechmark": "असमर्थित बेंचमार्क",
    "Unsupported media format": "असमर्थित मीडिया प्रारूप",
    "WARNING: maxThreadsPerMultiProcessor can not be detected": "चेतावनी: maxThreadsPerMultiProcessor का पता नहीं लगाया जा सकता",
    "WARNING: the detected photon number is more than what your have specified": "चेतावनी: पता चला फोटॉन संख्या आपके द्वारा निर्दिष्ट से अधिक है",
    "WARNING: the saved trajectory positions are more than what your have specified": "चेतावनी: सहेजी गई प्रक्षेपवक्र स्थितियाँ आपके द्वारा निर्दिष्ट से अधिक हैं",
    "WARNING: this MCX binary can not save partial path, please recompile mcx and make sure -D SAVE_DETECTORS is used by nvcc": "चेतावनी: यह MCX बाइनरी आंशिक पथ सहेज नहीं सकती, कृपया mcx को पुनः संकलित करें और सुनिश्चित करें कि nvcc द्वारा -D SAVE_DETECTORS का उपयोग किया गया है",
    "workload was unspecified for an active device": "सक्रिय डिवाइस के लिए कार्यभार निर्दिष्ट नहीं किया गया था",
    "you can not specify both interactive mode and config file": "आप इंटरैक्टिव मोड और कॉन्फ़िग फ़ाइल दोनों को एक साथ निर्दिष्ट नहीं कर सकते"
}),

MSTR(
{
    "_LANG_": "Русский",
    "_LOCALE_": "ru_RU",
    "_MCX_BANNER_":
) "\"" S_MAGENTA "###############################################################################\n\
#                    MCX (Monte Carlo eXtreme) -- CUDA                        #\n\
#    Авторское право (c) 2009–2025 Цяньцянь Фан <q.fang at neu.edu>           #\n\
#" S_BLUE "                https://mcx.space/  &  https://neurojson.io                  " S_MAGENTA "#\n\
#                                                                             #\n\
# Лаборатория вычислительной оптики и трансляционной визуализации (COTI) —    #\n\
#            " S_BLUE "http://fanglab.org " S_MAGENTA ", Биоинженерный факультет,                    #\n\
#  Северо-Восточный университет, Бостон, Массачусетс, США                     #\n\
###############################################################################\n\
# Разработка программного обеспечения MCX была поддержана NIH/NIGMS США       #\n\
# (грант R01-GM114365). Мы выражаем благодарность за поддержку.               #\n\
###############################################################################\n\
# Открытый научный код и переиспользуемые научные данные критически важны для #\n\
# современного научного прогресса. Команда MCX активно поддерживает открытую  #\n\
# науку, разработав JSON-формат ввода/вывода с поддержкой NIH.                #\n\
#                                                                             #\n\
# Посетите наш портал открытых данных NeuroJSON.io (" S_BLUE "https://neurojson.io" S_MAGENTA "),    #\n\
# и делитесь своими научными данными в удобном и переиспользуемом формате JSON#\n" S_RESET "\"," MSTR(
    "absorbed": "Общая доля поглощения",
    "after encoding": "Коэффициент сжатия после кодирования",
    "A CUDA-capable GPU is not found or configured": "Не найдено или не настроено GPU с поддержкой CUDA",
    "Built-in benchmarks": "Встроенные тесты",
    "Built-in languages":  "Встроенные языки",
    "code name": "Кодовое имя",
    "Command option": "Параметр команды",
    "compiled by nvcc": "Скомпилировано с помощью nvcc",
    "compiled with": "Скомпилировано с",
    "compressing data": "Сжатие данных",
    "compression ratio": "Коэффициент сжатия",
    "data normalization complete": "Нормализация данных завершена",
    "detected": "Обнаружено",
    "Downloading simulations from NeuroJSON.io (https://neurojson.org/db/mcx": "Загрузка симуляций с NeuroJSON.io (https://neurojson.org/db/mcx)",
    "ERROR: No CUDA-capable GPU device found": "ОШИБКА: не найдено устройство GPU с поддержкой CUDA",
    "ERROR: Specified GPU ID is out of range": "ОШИБКА: Указанный идентификатор GPU вне допустимого диапазона",
    "generating": "Генерация",
    "GPU ID can not be more than 256": "Идентификатор GPU не может превышать 256",
    "GPU ID must be non-zero": "Идентификатор GPU не может быть нулевым",
    "GPU Information": "Информация о GPU",
    "GPU memory can not hold all time gates, disabling normalization to allow multiple runs": "Память GPU не может вместить все временные окна, нормализация отключена для многократного запуска",
    "incomplete input": "Неполный ввод",
    "init complete": "Инициализация завершена",
    "initializing streams": "Инициализация потоков",
    "input domain is 2D, the initial direction can not have non-zero value in the singular dimension": "В 2D-вводе начальное направление не может иметь ненулевое значение в одномерном измерении",
    "invalid json fragment following --json": "Неверный JSON-фрагмент после --json",
    "Jacobian output is only valid in the reply mode. Please give an mch file after '-E'.": "Вывод Якобиана доступен только в режиме воспроизведения. Укажите файл .mch после '-E'.",
    "json fragment is expected after --json": "После --json ожидается JSON-фрагмент",
    "json shape constructs are expected after -P": "После -P ожидаются JSON-описания формы",
    "kernel complete": "Ядро завершено",
    "launching MCX simulation for time window": "Запуск симуляции MCX для временного окна",
    "loss due to initial specular reflection is excluded in the total": "Потери из-за начального зеркального отражения не учитываются в общем результате",
    "MCX Revision": "Версия MCX",
    "MCX simulation speed": "Скорость симуляции MCX",
    "No GPU device found": "GPU устройство не найдено",
    "normalization factor": "Коэффициент нормализации",
    "normalization factor for detector": "Коэффициент нормализации для детектора",
    "normalizing raw data ...": "Нормализация необработанных данных ...",
    "photons": "Фотоны",
    "please use the -H option to specify a greater number": "Пожалуйста, используйте параметр -H для указания большего значения",
    "please use the --maxjumpdebug option to specify a greater number": "Пожалуйста, используйте параметр --maxjumpdebug для указания большего значения",
    "random numbers": "Случайные числа",
    "requesting shared memory": "Запрос общей памяти",
    "respin number can not be 0, check your -r/--repeat input or cfg.respin value": "Число повторов не может быть 0. Проверьте параметры -r/--repeat или значение cfg.respin",
    "retrieving fields": "Получение полей",
    "retrieving random numbers": "Получение случайных чисел",
    "saved trajectory positions": "Сохранённые позиции траекторий",
    "saving data complete": "Сохранение данных завершено",
    "saving data to file": "Сохранение данных в файл",
    "seed length": "Длина зерна (seed)",
    "seeding file is not supported in this binary": "Файл инициализации не поддерживается в данной версии",
    "simulated":  "Смоделировано",
    "simulation run#": "Запуск симуляции #",
    "source": "Источник",
    "the specified output data format is not recognized": "Указанный формат выходных данных не поддерживается",
    "the specified output data type is not recognized": "Указанный тип выходных данных не поддерживается",
    "total simulated energy": "Общая энергия симуляции",
    "transfer complete": "Передача завершена",
    "unable to save to log file, will print from stdout": "Не удалось сохранить в лог-файл, вывод будет через стандартный поток",
    "unknown short option": "Неизвестный короткий параметр",
    "unknown verbose option": "Неизвестный длинный параметр",
    "unnamed": "Безымянный",
    "Unsupported bechmark": "Неподдерживаемый тест",
    "Unsupported media format": "Неподдерживаемый формат среды",
    "WARNING: maxThreadsPerMultiProcessor can not be detected": "ПРЕДУПРЕЖДЕНИЕ: maxThreadsPerMultiProcessor не может быть определён",
    "WARNING: the detected photon number is more than what your have specified": "ПРЕДУПРЕЖДЕНИЕ: обнаружено больше фотонов, чем было указано",
    "WARNING: the saved trajectory positions are more than what your have specified": "ПРЕДУПРЕЖДЕНИЕ: количество сохранённых траекторий превышает указанное значение",
    "WARNING: this MCX binary can not save partial path, please recompile mcx and make sure -D SAVE_DETECTORS is used by nvcc": "ПРЕДУПРЕЖДЕНИЕ: данная версия MCX не поддерживает сохранение частичных путей. Пожалуйста, перекомпилируйте с флагом -D SAVE_DETECTORS",
    "workload was unspecified for an active device": "Нагрузка не была указана для активного устройства",
    "you can not specify both interactive mode and config file": "Нельзя одновременно указать интерактивный режим и конфигурационный файл"
}),

MSTR(
{
    "_LANG_": "Português",
    "_LOCALE_": "pt_BR",
    "_MCX_BANNER_": 
) "\"" S_MAGENTA "###############################################################################\n\
#                          Monte Carlo Extremo (MCX) -- CUDA                  #\n\
#           Direitos autorais (c) 2009-2025 Qianqian Fang <q.fang at neu.edu> #\n\
#" S_BLUE "                https://mcx.space/  &  https://neurojson.io                  " S_MAGENTA "#\n\
#                                                                             #\n\
# Laboratório de Óptica Computacional e Imagem Translacional (COTI) - " S_BLUE "http://fanglab.org " S_MAGENTA "#\n\
# Departamento de Bioengenharia, Northeastern University, Boston, MA, EUA     #\n\
###############################################################################\n\
# O desenvolvimento do MCX foi financiado pelo NIH/NIGMS dos EUA (R01-GM114365), agradecemos o apoio\n\
###############################################################################\n\
# Código aberto para pesquisa científica e dados reutilizáveis são essenciais para a ciência moderna.\n\
# A equipe MCX apoia ciência aberta e, com financiamento do NIH, desenvolveu formato JSON para entrada e saída.\n\
#                                                                             #\n\
# Visite nosso portal de dados abertos NeuroJSON.io (" S_BLUE "https://neurojson.io" S_MAGENTA ")    #\n\
# e convidamos você a compartilhar seus dados científicos usando o formato JSON #\n\
# simples e reutilizável e nosso site gratuito.           #\n" S_RESET "\"," MSTR(
    "absorbed": "proporção total absorvida",
    "after encoding": "Taxa de compressão após codificação",
    "A CUDA-capable GPU is not found or configured": "GPU compatível com CUDA não encontrada ou configurada",
    "Built-in benchmarks": "Testes internos",
    "Built-in languages":  "Idiomas integrados",
    "code name": "nome da versão",
    "Command option": "opção de comando",
    "compiled by nvcc": "compilado com nvcc",
    "compiled with": "configuração da compilação",
    "compressing data": "compactando dados",
    "compression ratio": "taxa de compressão",
    "data normalization complete": "normalização dos dados concluída",
    "detected": "detectado",
    "Downloading simulations from NeuroJSON.io (https://neurojson.org/db/mcx": "Baixando simulações de NeuroJSON.io (https://neurojson.org/db/mcx)",
    "ERROR: No CUDA-capable GPU device found": "ERRO: Nenhum dispositivo GPU compatível com CUDA encontrado",
    "ERROR: Specified GPU ID is out of range": "ERRO: ID de GPU especificado está fora do intervalo",
    "generating": "gerando",
    "GPU ID can not be more than 256": "ID da GPU não pode ser maior que 256",
    "GPU ID must be non-zero": "ID da GPU não pode ser zero",
    "GPU Information": "Informações da GPU",
    "GPU memory can not hold all time gates, disabling normalization to allow multiple runs": "Memória da GPU não suporta todas as janelas de tempo, desabilitando normalização para permitir múltiplas execuções",
    "incomplete input": "entrada incompleta",
    "init complete": "inicialização concluída",
    "initializing streams": "inicializando GPU",
    "input domain is 2D, the initial direction can not have non-zero value in the singular dimension": "domínio de entrada é 2D, a direção inicial não pode ter valor diferente de zero na dimensão singular",
    "invalid json fragment following --json": "fragmento JSON inválido após --json",
    "Jacobian output is only valid in the reply mode. Please give an mch file after '-E'.": "Saída Jacobiana válida somente no modo replay. Por favor, forneça arquivo .mch após '-E'.",
    "json fragment is expected after --json": "esperado fragmento JSON após --json",
    "json shape constructs are expected after -P": "esperadas construções de forma JSON após -P",
    "kernel complete": "execução do kernel GPU concluída",
    "launching MCX simulation for time window": "iniciando simulação MCX para janela temporal",
    "loss due to initial specular reflection is excluded in the total": "perda por reflexão especular inicial excluída do total",
    "MCX Revision": "Revisão MCX",
    "MCX simulation speed": "velocidade da simulação MCX",
    "No GPU device found": "Nenhum dispositivo GPU encontrado",
    "normalization factor": "fator de normalização",
    "normalization factor for detector": "fator de normalização para detector",
    "normalizing raw data ...": "normalizando dados brutos...",
    "photons": "fótons",
    "please use the -H option to specify a greater number": "por favor, use a opção -H para especificar um número maior",
    "please use the --maxjumpdebug option to specify a greater number": "por favor, use a opção --maxjumpdebug para especificar um número maior",
    "random numbers": "números aleatórios",
    "requesting shared memory": "solicitando memória compartilhada",
    "respin number can not be 0, check your -r/--repeat input or cfg.respin value": "número de repetições não pode ser 0; verifique entrada -r/--repeat ou valor cfg.respin",
    "retrieving fields": "recuperando matrizes 3D de saída",
    "retrieving random numbers": "recuperando números aleatórios gerados",
    "saved trajectory positions": "posições da trajetória salvas",
    "saving data complete": "salvamento de dados concluído",
    "saving data to file": "salvando dados em arquivo",
    "seed length": "tamanho da semente (4 bytes)",
    "seeding file is not supported in this binary": "arquivo de semente não suportado neste binário",
    "simulated":  "simulado",
    "simulation run#": "execução da simulação #",
    "source": "fonte",
    "the specified output data format is not recognized": "formato de dados de saída especificado não reconhecido",
    "the specified output data type is not recognized": "tipo de dados de saída especificado não reconhecido",
    "total simulated energy": "energia total simulada",
    "transfer complete": "transferência concluída",
    "unable to save to log file, will print from stdout": "não foi possível salvar arquivo de log, será impresso na saída padrão",
    "unknown short option": "opção curta desconhecida",
    "unknown verbose option": "opção longa desconhecida",
    "unnamed": "sem nome",
    "Unsupported bechmark": "benchmark não suportado",
    "Unsupported media format": "formato de mídia não suportado",
    "WARNING: maxThreadsPerMultiProcessor can not be detected": "AVISO: maxThreadsPerMultiProcessor não pode ser detectado",
    "WARNING: the detected photon number is more than what your have specified": "AVISO: número de fótons detectados é maior que o especificado",
    "WARNING: the saved trajectory positions are more than what your have specified": "AVISO: posições da trajetória salvas são maiores que o especificado",
    "WARNING: this MCX binary can not save partial path, please recompile mcx and make sure -D SAVE_DETECTORS is used by nvcc": "AVISO: este binário MCX não pode salvar caminho parcial, recompile MCX e certifique-se de usar -D SAVE_DETECTORS com nvcc",
    "workload was unspecified for an active device": "carga de trabalho não especificada para dispositivo ativo",
    "you can not specify both interactive mode and config file": "não é possível especificar modo interativo e arquivo de configuração simultaneamente"
})

};
