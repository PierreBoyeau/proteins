###################################################
# Makefile (*.ccをコンパイルする場合)
# コンパイルしたいソースがあるフォルダと同じところに置く
# ディレクトリ階層図
# this
# |--- *.cc          (コンパイルしたいソースファイル)
# |--- Makefile
# |--- train        (生成される実行ファイル)
# |--- obj            (生成されるフォルダ)
#      |--- *.o       (生成されるオブジェクトファイル)
#      |--- *.d       (生成される依存関係ファイル)
#
# 実行コマンド
# make         (更新されたファイルのみコンパイル)
# make all     (クリーンしてビルド)
# make clean   (クリーン)
###################################################
AR       = ar
ARFLAGS  = rus

CXX      = g++

# fopenmpオプションをつけると並列化処理用になる
CXXFLAGS = -g -MMD -MP -Wall -Wextra -Winit-self -Wno-unused-parameter -fopenmp -std=c++11 -O3

RM       = rm -f
LDFLAGS  = -fopenmp
LIBS     =
INCLUDE  = -I../include
#実行ファイル
TARGET   = ./train
#オブジェクトファイル保存フォルダ
OBJDIR   = ./obj
#ソースファイル
SOURCES  = $(wildcard *.cc)
#オブジェクトファイル
OBJECTS  = $(addprefix $(OBJDIR)/, $(SOURCES:.cc=.o))
#依存関係ファイル
DEPENDS  = $(OBJECTS:.o=.d)
#実行ファイル作成
$(TARGET): $(OBJECTS) $(LIBS)
	$(CXX) -o $@ $^ $(LDFLAGS)
#ソースファイルをコンパイル
$(OBJDIR)/%.o: %.cc
	@if [ ! -d $(OBJDIR) ];\
	then echo "mkdir -p $(OBJDIR)";mkdir -p $(OBJDIR);\
	fi
	$(CXX) $(CXXFLAGS) $(INCLUDE) -o $@ -c $<

#クリーンしてビルド
.PHONY: all
all: clean $(TARGET)

#クリーン
.PHONY:clean
clean:
	$(RM) $(OBJECTS) $(DEPENDS) $(TARGET)

-include $(DEPENDS)
