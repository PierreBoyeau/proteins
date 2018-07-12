#include "prefixspan.h"

void PrefixSpan::printTree(string aFilename){
	ofstream tFile(aFilename);
	uint tCnt = 0;
	for(uint i = 0;i < Active.size();i++){
		if(Active[i]->w!=0){
			tCnt++;
		}
	 }
	tFile << "Size of data :"<< mN << '\n';
	tFile << "Size of tree :"<<tree.size() << '\n';
	tFile << "Size of Active :"<<tCnt << '\n';
	tFile << "pattern,supportSum,addλ,w,list[id:sup]" << '\n';
	//木丸ごと書き込む時はこっち
	// for(auto itr = tree.begin();itr != tree.end();++itr){
	// 		 tFile << itr->patternStr << "," <<itr->supportSum << "," << itr->w << "," << itr->isLeaf << '\n';
	// }
	// アクティブだけ
	for(uint i = 0;i < Active.size();i++){
		if(Active[i]->w!=0){
			tFile << Active[i]->patternStr << "," <<Active[i]->supportSum << "," <<Active[i]->addLambda<<","<< Active[i]->w<<",[";
				for(uint j=0;j<Active[i]->support.size();j++)
					tFile <<Active[i]->x[j]<<":"<<Active[i]->support[j]<<",";
				tFile <<"]" <<'\n';
		}
	 }
}

void PrefixSpan::add_history(uint lambda){

	for(auto& it : tree){
		if(it.w != 0){
			if(it.addLambda == -1){
				it.addLambda = lambda;
			}
		} else {
			it.addLambda = -1;
		}
	}
}

uint PrefixSpan::is_subsequence(const vector<Event> aSeq1, const vector<Event> aSeq2){
	vector<Event> tShort_seq;
	vector<Event> tLong_seq;
	uint tSub_seq;

	if(aSeq1.size() < aSeq2.size()){
		tShort_seq = aSeq1;
		tLong_seq = aSeq2;
		tSub_seq = 1;
	} else if(aSeq1.size() > aSeq2.size()){
		tShort_seq = aSeq2;
		tLong_seq = aSeq1;
		tSub_seq = 2;
	} else {
		return 0;
	}

	uint tCount = 0;
	uint diff_size = tLong_seq.size() - tShort_seq.size();

	for(uint i = 0; i <= diff_size; ++i){
		for(uint it = 0; it < tShort_seq.size(); ++it){
			if(tShort_seq[it] == tLong_seq[it + i]){
				tCount++;
			}else{
				tCount = 0;
				break;
			}

			if(tCount == tShort_seq.size()){
				return tSub_seq;
			}
		}
	}


	return 0;
}

uint PrefixSpan::sum_items_pdb(const projectDB aPdb){
	uint tSum = 0;
	for(auto itr : aPdb){
		tSum += mTransaction[itr.first].size() - itr.second.eid - 1;
	}
	return tSum;

}


//for predict
int PrefixSpan::calcSup(const vector<Event> &aTransaction,const vector<Event> &aPattern) const{
		uint patSize = aPattern.size();
		uint k;
		int num = 0;
		int interval;
			
		//想定しているパターンよりデータが短い場合は0を返す
		if(aTransaction.size() < patSize) return num;

		for(uint i = 0; i<= aTransaction.size() - patSize; i++){

			interval = 0;

			k = 0;

			for(uint j = i; j < aTransaction.size(); j++){
					
					if(maxInterval >= 0 && interval  > maxInterval){
						break;
					}
					if(aTransaction[j] == aPattern[k]){
						k++;
						interval= -1;
						if(k == patSize){
							//0/1の場合
							if(supMode == 0){
								return 1;
							}
							k = 0;
							num++;
							i = j;
							break;
						}
					}
					interval++;
				}
			}
			return num;
	}

double PrefixSpan::calcSup(uint aId,vector<Event> aPattern){
  //pierre: compute support
  //pierre: ie if the value is binary True/False, or if we count patterns.
  switch (supMode) {
		case 1:
			return calcPat(aId,aPattern);
			break;
		default:
			return 1;
			break;
  }
}

int PrefixSpan::calcPat(uint aId,vector<Event> aPattern){
	uint patSize = aPattern.size();
	uint k;
	int num = 0;
	int interval;

	//想定しているパターンよりデータが短い場合は0を返す
	if(mTransaction[aId].size() < patSize) return num;

	for(uint i = 0; i <= mTransaction[aId].size() - patSize; i++){

		interval = 0;

		k = 0;
		for(uint j = i; j < mTransaction[aId].size(); j++){

				if(maxInterval >= 0 && interval  > maxInterval){
					break;
				}
				if(mTransaction[aId][j] == aPattern[k]){
					k++;
					interval= -1;
					if(k == patSize){
						k = 0;
						num++;
						i = j;
						break;
					}
				}
				interval++;
			}
		}
		return num;
}

//初期の木を構築している
//pierre: building initial tree
void PrefixSpan::init(const vector< vector<Event> > aTransaction,const vector<double> aY) {
	mTransaction = aTransaction;
	mY = aY;
	mEventSize = mTransaction[0][0].size();
	mN = mTransaction.size();  //pierre: nb of examples.
	if(mEventSize == 0){
			cout << "error:Event size is zero." << '\n';
			exit(1);
		}

		//nはデータ数
	if(mTransaction.empty()|| mY.empty()){
			cout << "error:Data or label is empty." << '\n';
		exit(1);
	}
	pattern.clear();
	tree.clear();
	Active.clear();
	mR.clear();

	//木の根をここで作成している
	//pierre: I have created a root of trees here

	map<Event, projectDB> root;
	map<Event,uint> dupCheck;
	//イベント間の間隔に制限を設けない時は初期ノードの重複を許さない
	if (maxInterval < 0){
	//ルートにはロードしたデータのイベントが全て入っている
		for (uint i = 0; i < mN; ++i) {
			//イベントごとにマッピングされる
			for (uint j = 0, size = mTransaction[i].size(); j < size; ++j) {
				if(dupCheck.find(mTransaction[i][j])==dupCheck.end()){
					dupCheck[mTransaction[i][j]]=0;
					//i jはi番目のデータのj番目の系列
					//pierre : i j is the j th sequence of the i th data

					root[mTransaction[i][j]].push_back(pair<uint, pos>(i, {j,0}));
					}
			}
			dupCheck.clear();
		}
	}else{
		for (uint i = 0; i < mN; ++i) {
			for (uint j = 0, size = mTransaction[i].size(); j < size; ++j) {
		root[mTransaction[i][j]].push_back(pair<uint, pos>(i, {j,0}));
	}
		}

	}
	pattern.clear();
	//root.begin()でイテレータの最初を取り出して，.end()で終わりを取り出せる

	//pierre: Unsure : We have located all possible patterns, now we store them.
	for (auto it = root.begin(), end = root.end(); it != end; ++it) {
		//pierre : Key or event is stored
		//キーつまりイベントが格納される
		pattern.push_back(it->first);
		Node node;
		node.pattern = pattern;
		node.patternStr = pat2str();
		node.supportSum = 0;
		node.pdb = it->second;
		node.w = 0;
		node.val = 0;
		node.isLeaf = true;
		node.sumItems = sum_items_pdb(it->second);

		uint oid = MAXVAL;
		//projectDBのサイズ回ループ
		for (uint i = 0, size = node.pdb.size(); i < size; ++i) {
			//idはレコード番号
			uint id = node.pdb[i].first;
			if (id != oid) {
				// node.supportSum++;
				double tSup = calcSup(id,pattern);
				node.supportSum += tSup;
				node.support.push_back(tSup);
				node.x.push_back(id);
			}
			oid = id;
		}

		tree.push_back(node);
		pattern.pop_back();
	}

}
/**
 * @fn
 * SPPC計算を行っている(type == 2の時)
 * @return 枝が切ることができればtrue
 */
bool PrefixSpan::calculate(Node &node) {
	if (node.supportSum < minsup) return true;

	node.val = 0;
	double p = 0, m = 0;
	//見ているノードpatternを持っているid数で回す
	for (uint i = 0; i < node.x.size(); ++i) {
		uint id = node.x[i];
		if (type == 1 || type == 2) { // svm
			if (mR[id] > 0) {
				//(初めは)vが1-ybとなる，これはλmaxのときの損失の値
				//@
				double val = alpha*mR[id]*mY[id]*node.support[i];
				node.val += val;
				(val > 0) ? p += val : m += val;
			}
		} else if (type == 3|| type == 4){ //lasso
				double val = alpha*mR[id]*node.support[i];
				node.val += val;
				(val > 0) ? p += val : m += val;
		} else if (type == 5 || type == 6){ //logistic
				double val = (mY[id] > 0) ? 1/(1+mR[id]) : 1/(1+mR[id])-1;
				val *= alpha*node.support[i];
				node.val += val;
				(val > 0) ? p += val : m += val;
		}

	}
	node.val = fabs(node.val);

	if (type == 1|| type == 3|| type == 5) { // get_maxnode
		if (max(p,-m) < Maxnode.val) return true;
	} else if (type == 2 || type == 4 || type == 6) { // safe_screening
		if (max(p,-m)+radius*sqrt(node.supportSum) < 1) return true;
	}

	return false;
}

/**
 * @fn
 * project_new内でのみ使用
 * サポートを数えている
 * @return 葉になることができればtrue
 */
bool PrefixSpan::calculate_new(Node &node) {
	uint oid = MAXVAL;
	double p = 0, m = 0;
	//見ているノードの持っているデータベースの長さ回まわす
	//pdb.sizeは同じデータ内であっても複数存在する可能性あり
	//init()でやっていることを行っているので新しいノードを展開した時に多分使用する
	for (uint i = 0, size = node.pdb.size(); i < size; ++i) {
		uint id = node.pdb[i].first;
		if (oid != id) {
			// node.supportSum++;
			double tSup = calcSup(id,node.pattern);
			node.supportSum += tSup;
			node.support.push_back(tSup);
			node.x.push_back(id);
			if (type == 1 || type == 2) { // svm
				if (mR[id] > 0) {
					double val = alpha*mR[id]*mY[id]*tSup;
					node.val += val;
					(val > 0) ? p += val : m += val;
				}
			} else if(type == 3 || type == 4){ // lasso
				double val = alpha*mR[id]*tSup;
				node.val += val;
				(val > 0) ? p += val : m += val;
			} else if(type == 5 || type == 6){ // logistic
				double val = (mY[id] > 0) ? 1/(1+mR[id]) : 1/(1+mR[id])-1;
				val *= alpha*tSup;
				node.val += val;
				(val > 0) ? p += val : m += val;
			}
		}
		oid = id;
	}

	node.val = fabs(node.val);

	if (minsup > node.supportSum) return true;

	if (type == 1 || type == 3 || type == 5) {
		if (max(p,-m) < Maxnode.val) return true;
	} else if (type == 2 || type == 4 || type == 6) {
		//SPPC計算 これより下は見る必要がないということ(ここが葉になる)
		if (max(p,-m)+radius*sqrt(node.supportSum) < 1) return true;
	}

	return false;
}

/**
 * @fn
 *今見ているpdbの要素全ての一つ後ろの要素をcountermapに突っ込んで
 *それら全てをノードとして生やすためにproject_newする
 *
 */
void PrefixSpan::project(projectDB &pdb) {
	// scan projected database
	if (pattern.size() < maxpat) {
		map<Event, projectDB> counter;

		if(maxInterval < 0){
			map<Event,uint> dupCheck;
			for (uint i = 0, size = pdb.size(); i < size; ++i) {
				uint id = pdb[i].first;
				uint trsize = mTransaction[id].size();
				//シーケンスの一つ隣のindexを取得
				uint j = pdb[i].second.eid+1;

				for (;j < trsize;j++) {
					// if(j<trsize){
			 		//最初に出てきたものだけを採用
					if(dupCheck.find(mTransaction[id][j]) == dupCheck.end()){
						dupCheck[mTransaction[id][j]] = 0;
						counter[mTransaction[id][j]].push_back(pair<uint, pos>(id, {j,0}));
					}
				}
				dupCheck.clear();
			}
		}else{
			map<uint,vector<uint> > dupCheck;
			for (uint i = 0, size = pdb.size(); i < size; ++i) {
				uint id = pdb[i].first;
				uint trsize = mTransaction[id].size();
				//シーケンスの一つ隣のindexを取得
				uint j = pdb[i].second.eid+1;
				int k = j;

				for (; j < trsize; j++) {
					//j-kは最大インターバルの計算
					if(j - k > maxInterval){break;}
					// if(j<trsize){
					//pdbに全く同じものが入らないようにする
					auto itr = find(dupCheck[id].begin(),dupCheck[id].end(),j);
					if(itr == dupCheck[id].end()){
						dupCheck[id].push_back(j);
						counter[mTransaction[id][j]].push_back(pair<uint, pos>(id, {j,0}));
					}
				}
			}
		}

		// project: next event
		for (auto it = counter.begin(), end = counter.end(); it != end; ++it) {
			pattern.push_back(it->first);
			project_new(it->second);
			pattern.pop_back();
		}
	}

}

/**
 * @fn
 *新しいノードを作成する
 *それが枝切りできるかチェックしてできるなら葉とする
 *できないなら木にこのノードを追加
 *UBを計算して展開すべきものかを見極める
 *葉にぶつかるまでprojectを掘る
 */
void PrefixSpan::project_new(projectDB &pdb) {
	Node node;
	node.pattern = pattern;
	node.patternStr = pat2str();
  	node.supportSum = 0;
	node.pdb = pdb;
	node.w = 0;
	node.val = 0;
	node.sumItems = sum_items_pdb(pdb);


	//ここでCloSpanチェック
	//tree全てのnodeと新しいnodeを比べて，包含関係にあってPDBが同じものがないかチェックする
	//PDBはsumItemsが同じかどうかでチェックする
	if(mCloSpan == 1){
		for(auto it : tree){
			if(node.sumItems == it.sumItems){
				if(is_subsequence(node.pattern, it.pattern) == 1){
//				cout << "calculate_new node:" << node.patternStr << ", it:" << it.patternStr << endl;
					return;
				}
			}
		}
	}


	//新しく作ったnodeが葉かどうかをしらべる
	bool flag = calculate_new(node);
	if (flag) {
		//実質ここでしかisLeafをtrueにできない
		node.isLeaf = true;
		tree.push_back(node);
		return;
	}

	node.isLeaf = false;
	//リストの最後にNodeを挿入
	treeIter current = tree.insert(tree.end(), node);

	if (type == 1 || type == 3 || type == 5) { // get_maxnode
		if (node.val > Maxnode.val) {
			Maxnode = node;
		}
	} else if (type == 2|| type == 4 || type == 6) { // safe_screening
		//SPPC計算
		double score = node.val+radius*sqrt(node.supportSum);
		if (score >= 1) {
			//closed sequential patternかどうかのフラグ
			bool tClose_flag = true;
			//ここで同一サポートかつより長いパターンを残すようにするチェックを挟む
			for(uint i = 0; i < Active.size(); ++i){

				if(Active[i]->supportSum == node.supportSum){
					uint tSub = is_subsequence(Active[i]->pattern, node.pattern);
					if(tSub == 1){
						Active.erase(Active.begin() + i);
						i--;
						break;
					} else if(tSub == 2){
						tClose_flag = false;
						break;
					}
				}
			}

			if(tClose_flag){
				Active.push_back(current);
			}
		}
		//if (score >= 1)で漏れたやつは切れないけどnonアクティブ あるかは不明
	}

	project(pdb);
}
/**
 * @fn
 *こいつを使うことで根から枝を伸ばし切る
 *
 */
void PrefixSpan::get_maxnode(const vector<double> &_v, int solver) {
	switch (solver) {
		case 1: // svm
			type = 1;
			break;

		case 2: //lasso
			type = 3;
			break;
		case 3: //logistic
			type = 5;
			break;
		default:
			cerr << "Error: unknown solver type" << endl;
			exit(1);
			break;
	}
	mR = _v;
	alpha = 1;
	Maxnode.val = 0;
	pattern.clear();

	vector<treeIter> Leafs;
	for (treeIter it = tree.begin(), end = tree.end(); it != end; ++it) {

		bool flag = calculate(*it);
		if (flag) {
			continue;
		} else {
			// 最大値更新
			if (it->val > Maxnode.val) Maxnode = *it;
			// Leafなら
			if (it->isLeaf) {
				it->isLeaf = false;
				Leafs.push_back(it);

			}
		}

	}

   //Leafsがまだ伸ばせる枝リストであるのでそこを伸ばし切る
	for (uint i = 0, size = Leafs.size(); i < size; ++i) {
		pattern = Leafs[i]->pattern;
		project(Leafs[i]->pdb);
	}

}

//残差，α，半径を用いて枝切りできるかを調べる
void PrefixSpan::safe_screening(vector<double> &_v, double _alpha, double _radius, int solver) {
	switch (solver) {
	case 1: // svm
		type = 2;
		break;
	case 2: // lasso
		type = 4;
		break;
	case 3: //logistic
		type = 6;
		break;
	default:
		cerr << "Error: unknown solver type" << endl;
		exit(1);
		break;
	}
	mR = _v;
	alpha = _alpha;
	radius = _radius;
	pattern.clear();
	Active.clear();

	vector<treeIter> Leafs;
	for (treeIter it = tree.begin(), end = tree.end(); it != end; ++it) {

		bool flag = calculate(*it);
		//UB(j) uj部分が最大値．
		if (flag) {
			it->w = 0;
			continue;
		} else {
			//uj部分が和の絶対値で少し小さい
			double score = it->val+radius*sqrt(it->supportSum);
			if (score >= 1) {
				//closed sequential patternかどうかのフラグ
				bool tClose_flag = true;
				//ここで同一サポートかつより長いパターンを残すようにするチェックを挟む
				for(uint i = 0; i < Active.size(); ++i){

					if(Active[i]->supportSum == it->supportSum){
						uint tSub = is_subsequence(Active[i]->pattern, it->pattern);
						if(tSub == 1){
							Active.erase(Active.begin() + i);
							i--;
							break;
						}else if(tSub == 2){
							tClose_flag = false;
							break;
						}
					}
				}

				if(tClose_flag){
					Active.push_back(it);
				}

			} else {
				it->w = 0;
			}

			if (it->isLeaf) {
				it->isLeaf = false;
				Leafs.push_back(it);
			}
		}
	}

	for (uint i = 0, size = Leafs.size(); i < size; ++i) {
		pattern = Leafs[i]->pattern;
		project(Leafs[i]->pdb);
	}

}
