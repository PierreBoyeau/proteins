#ifndef PREFIXSPAN_H
#define PREFIXSPAN_H

#include <iostream>
#include <fstream>
#include <cmath>
#include <map>
#include <vector>
#include <list>
#include <string>
#include <sstream>
#include <algorithm>

#include "database.h"

using namespace std;

using uint = unsigned int;
using uchar = unsigned char;

class PrefixSpan{
private:
	struct pos {
		//系列の何番目かを保存している
		uint eid;
		//こいつはどこで使っているのか？
		uint did;
	};
	//projectDBのuintはid，つまりデータとして何番目にいるかを保存している
	using projectDB = vector< pair<uint, pos> >;
	using Event = vector<uint>;

	struct Node {
		vector<Event> pattern;
		string patternStr;
		double supportSum;
		// uint support;
		vector<double> support;
		projectDB pdb;
		vector<uint> x;
		double w;
		double val;
		bool isLeaf;
		int addLambda = -1;
		uint sumItems; //PDB内の総アイテム数（CloSpan用）
	};

	const uint MAXVAL = 0xffffffff;
	uint mEventSize;
	// uint minsup;
	double minsup;
	uint maxpat;
	int maxInterval;
	uint supMode;
	uint mCloSpan;


	int type;
	vector<double> mR;
	double alpha;
	double radius;

 //ここにファイルから読み込んだデータが入る
	vector< vector<Event> > mTransaction;
	vector<Event> pattern;
	list<Node> tree;
	using treeIter = list<Node>::iterator;

	/**
	 * @fn
	 * 数字として管理しているパターンを文字列にする
	 * @return あるノードにおけるパターンを文字列で出力
	 *mEventSizeとは１；１；１だったら３ということ
	 */
	string pat2str(void) {
		uint i;
		uint size = pattern.size();
		stringstream ss;
		for (i = 0; i < size; ++i) {
			ss << (i ? " " : "");
			for (uint j = 0; j < mEventSize; j++) {
				if (pattern[i][j] == MAXVAL) ss << "*";
				else ss << pattern[i][j];
				ss << (j+1 < mEventSize ? ":" : "");
			}
		}
		return ss.str();
	}

	bool calculate(Node &node);
	bool calculate_new(Node &node);
	void project(projectDB &pdb);
	void project_new(projectDB &pdb);
	/**
	*あるデータの中の任意のパターンのサポートを数える
	*mode:0,(1,2,3以外の数字が入った時) supportは１レコード1まで
	*mode:1 単純なパターンの数え上げ，1レコードにいくらでも可能
	*@return supprt数
	**/
 	double calcSup(uint aId,vector<Event> aPattern);

	/*
	*trainTransaction[aId]であるデータ内からパターンがaPatternであるものを数え上げる
	*@return 数え上げた数
	*/
	int calcPat(uint aId,vector<Event> aPattern);

	/*
	 * 入力された系列が包含関係かどうか調べる．
	 * 包含関係なら1番目か2番目のどちらが短いか（subsequence）を返す．
	 * @return 1 or 2 or 0(包含関係ではない or 全く同一系列)
	 * @author takuto
	 */
	uint is_subsequence(const vector<Event> aSeq1, const vector<Event> aSeq2);

	/*
	 * CloSpan用
	 * @return PDB内の総アイテム数
	 * @author takuto
	 */
	uint sum_items_pdb(const projectDB aPdb);


public:

	uint mN;
	vector<double> mY;
	Node Maxnode;
	vector<treeIter> Active;


 //コンストラクタ
	PrefixSpan(double _minsup, uint _maxpat, int _interval,uint _supMode, uint _CloSpan) {
		minsup = _minsup;
		maxpat = _maxpat;
		maxInterval = _interval;
		supMode = _supMode;
		mCloSpan = _CloSpan;
	};

	void init(const vector< vector<Event> > aTransaction,const vector<double> aY);
	void get_maxnode(const vector<double> &_v, int solver);
	void safe_screening(vector<double> &_v, double _alpha, double _radius, int solver);
	void printTree(string aFilename);
	void add_history(uint lambda);
	//transactionの中のpattern数を数える
	int calcSup(const vector<Event> &aTransaction,const vector<Event> &aPattern) const;
};

#endif
