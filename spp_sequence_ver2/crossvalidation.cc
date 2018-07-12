#include "crossvalidation.h"


uint CrossValidation::select(const vector<double> &aSolutionPath,const Database &aDatabase,PrefixSpan &aPrefix,LearnerSPP &aLearner,const vector<uint> &aOptions){
	uint tK = aOptions[0];
	uint tAve = aOptions[1];
	uint tLossType = aOptions[2];
	//各λの正解数を保持
	//pierre: Keep number of correct answers for each λ
	vector<double> tCnt(aSolutionPath.size() - 1, 0);

	//全データからテストデータと学習データのk分割をこしらえる
	vector<vector<Event> > tTransaction = aDatabase.get_transaction();
	vector<double> tY = aDatabase.get_y();
	uint tN = tY.size();
	if(tN < tK){
		cout << "error:CV,n<k" << '\n';
		exit(1);
	}

	cout << "CV now ";

	for(uint i = 0; i < tAve; ++i){
		if((i + 1) % 5 == 0){
			cout << i + 1 << flush;
		}else{
			cout << "." << flush;
		}

		//1.データがk分割すると何個づつになるかを計算する
		//例えば 19 = [4,5,5,5]
		for(uint j = 0; j < tK; ++j){
			mNumDiv.push_back((tN + j) / tK);
		}

		//index をシャッフル
		for(uint j = 0; j < tN; ++j){
			mTrainIdx.push_back(j);
		}

		for(uint j = 0; j < tN; ++j){
			uint k = j + (rand() % (tN - j));
			swap(mTrainIdx[j], mTrainIdx[k]);
		}

		for(uint j = 0; j < mNumDiv.back(); ++j){
			mTestIdx.push_back(mTrainIdx.back());
			mTrainIdx.pop_back();
		}
		mNumDiv.pop_back();

		//k-foldの結果がtCntに入る
		k_fold(aSolutionPath, aPrefix, aLearner, tTransaction, tY, tK, tCnt,tLossType);

		mNumDiv.clear();
		mTrainIdx.clear();
		mTestIdx.clear();
	}

	vector<double>::iterator tIter;
	cout << '\n';
	switch (tLossType) {
		// L2SVM
		case 1:
		for(uint i = 0; i < tCnt.size(); ++i){
			cout << "λ[" << i + 1 << "]=" << aSolutionPath[i + 1] << " " << tCnt[i] << "/" << aOptions[1] * tN << "=" << tCnt[i] / (double) (aOptions[1] * tN) << '\n';
		}

		//最大値選び
		tIter = max_element(tCnt.begin(), tCnt.end());
		break;
		//Lasso
		case 2:
		for(uint i = 0; i < tCnt.size(); ++i){
			cout << "λ[" << i + 1 << "]=" << aSolutionPath[i + 1] << " RMSE:" <<sqrt(tCnt[i] / (double) (aOptions[1] * tN)) << '\n';
		}

		//最小値選び
		tIter = min_element(tCnt.begin(), tCnt.end());
		break;
		default:
		std::cout << "error:CV output" << '\n';
	}


	return distance(tCnt.begin(), tIter) + 1;
}

void CrossValidation::calc_accuracy(const vector<vector<double> > &aYHats,const vector<double> &aY,vector<double> &aCnt,const uint &aLossType){
  if(aCnt.size()!=aYHats.size()){
    cout << "error:calc_accuracy" <<aCnt.size()<<":"<<aYHats.size()<< '\n';
    exit(1);
  }

  for(uint j=0;j<aYHats.size();++j){
	//pierre: aYHats has shape (nb_Lambdas, nb_examples)
	
    for(uint k=0;k<aY.size();++k){

       if(aLossType==1 && aY[k]*aYHats[j][k]>0){
         // for L2SVM:当たっている数
         aCnt[j]++;
       }else if(aLossType==2){
         // foe LASSO;2乗損失
				 aCnt[j] += (aY[k]-aYHats[j][k])*(aY[k]-aYHats[j][k]);
			 }
    }
  }
}


void CrossValidation::calc_precision(const vector<vector<double> > &aYHats,const vector<double> &aY,vector<double> &aCnt,const uint &aLossType){
  	if(aLossType==2){
		cout << "error:calc_precision: precision defined only for classification tasks"<< '\n';
		exit(1);
    }
	for(uint j=0;j<aYHats.size();++j){
		double true_positive = 0;
		double false_positive = 0;
		for(uint i=0;i<aY.size();++i){
			if(aY[i]>0 && aYHats[j][i]>0)
			{
				true_positive++;
			}
			
			else if (aY[i]<0 && aYHats[j][i]>0)
			{
				false_positive++;
			}
		}
		aCnt[j] = (double) true_positive / (true_positive+false_positive);
	}
}

void CrossValidation::calc_recall(const vector<vector<double> > &aYHats,const vector<double> &aY,vector<double> &aCnt,const uint &aLossType){
	if(aLossType==2){
		cout << "error:calc_recall: recall defined only for classification tasks"<< '\n';
		exit(1);
  	}
	for(uint j=0;j<aYHats.size();++j){
		double true_positive = 0;
		double all_positive = 0;
		for(uint i=0;i<aY.size();++i){
			if(aY[i]>0)
			{
				all_positive++;
				if(aYHats[j][i]>0)
				{
					true_positive++;
				}
			}
		}
		aCnt[j] = (double) true_positive / all_positive;
	}
}


void CrossValidation::next_train_test(){
	if(mNumDiv.empty()){
		cout << "error:CV next_train_test" << '\n';
		exit(1);
	}
	vector<uint> tTmp;
	//tTmp にmTestIdxをコピー
	copy(mTestIdx.begin(), mTestIdx.end(), back_inserter(tTmp) );
	mTestIdx.clear();
    //train から次の数抜いてtestに突っ込む
  	for(uint i=0;i<mNumDiv.back();++i){
		mTestIdx.push_back(mTrainIdx.back());
		mTrainIdx.pop_back();
  	}
	mNumDiv.pop_back();
	//tmpに残っているtrainをタス
	copy(mTrainIdx.begin(),mTrainIdx.end(),back_inserter(tTmp));
	mTrainIdx.clear();
	//tempをtrainにする終わり
	copy(tTmp.begin(), tTmp.end(), back_inserter(mTrainIdx));
}

void CrossValidation::k_fold(const vector<double> &aSolutionPath, PrefixSpan &aPrefix, 
								LearnerSPP &aLearner, const vector<vector<Event>> &aTransaction, const vector<double> &aY, 
								const uint &aK, vector<double> &aCnt,const uint &aLossType){

	vector<uint> tSVMOption = { (uint) aSolutionPath.size() };

	for(uint i = 0; i < aK; ++i){
		vector<vector<Event> > tTrainTransaction(mTrainIdx.size());
		vector<double> tTrainY(mTrainIdx.size());
		for(uint j = 0; j < mTrainIdx.size(); ++j){
			tTrainTransaction[j] = aTransaction[mTrainIdx[j]];
			tTrainY[j] = aY[mTrainIdx[j]];
		}

		vector<vector<Event> > tTestTransaction(mTestIdx.size());
		vector<double> tTestY(mTestIdx.size());
		for(uint j = 0; j < mTestIdx.size(); ++j){
			tTestTransaction[j] = aTransaction[mTestIdx[j]];
			tTestY[j] = aY[mTestIdx[j]];
		}

		aPrefix.init(tTrainTransaction, tTrainY);
		//pierre: question prediction computed on all data or only on training?
		vector<vector<double> > tYHats = aLearner.get_all_predict(aPrefix, aSolutionPath, tTestTransaction, tSVMOption);
		calc_accuracy(tYHats, tTestY, aCnt,aLossType);
		//test trainの切り替え
		if(i != aK - 1){
			next_train_test();
		}
	}

}
