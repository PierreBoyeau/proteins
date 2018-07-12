#ifndef CROSSVALIDATION_H
#define CROSSVALIDATION_H

#include "modelselector.h"
#include <random>

//learnerspp
#include "svmspp.h"
#include "lassospp.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;

using uint = unsigned int;
using Event = vector<uint>;

class CrossValidation: public ModelSelector{
private:
	vector<uint> mTrainIdx;
	vector<uint> mTestIdx;
	vector<uint> mNumDiv;

public:
	CrossValidation(){};
	void calc_accuracy(const vector<vector<double> > &aYHats, const vector<double> &aY, vector<double> &aCnt,const uint &aLossType);
	void calc_precision(const vector<vector<double> > &aYHats, const vector<double> &aY, vector<double> &aCnt,const uint &aLossType);
	void calc_recall(const vector<vector<double> > &aYHats, const vector<double> &aY, vector<double> &aCnt,const uint &aLossType);

	//pierre: specific for Lambda CV!
	void next_train_test();

	//テンプレートでaLearnerの型を抽象化してます． aLearnerの型によって処理が変わるなら具体的な型名をつけたものを実装としてかけばオーバーロードされます．
	//pierre : We abstract aLearner type in template. If processing varies depending on the type of aLearner,
	//pierre : it will be overloaded by implementing a concrete type name added as an implementation
	template<class T> uint select(const vector<double> &aSolutionPath, const Database &aDatabase, PrefixSpan &aPrefix, T &aLearner, const vector<uint> &aOptions){
		uint tK = aOptions[0];
		uint tAve = aOptions[1];
		uint tLossType = aOptions[2];

		//全データからテストデータと学習データのk分割をこしらえる
		vector<vector<Event> > tTransaction = aDatabase.get_transaction();
		vector<double> tY = aDatabase.get_y();
		uint tN = tY.size();
		if(tN < tK){
			cout << "error:CV,n<k" << '\n';
			exit(1);
		}

		//各λの正解数を保持
		//pierre: Keep number of correct answers for each λ
		vector<double> tCnt(aSolutionPath.size() - 1, 0);
		vector<double> tPrecisions(aSolutionPath.size() - 1, 0);
		vector<double> tRecalls(aSolutionPath.size() - 1, 0);
		


		//1.データがk分割すると何個づつになるかを計算する
		//pierre: 1. Calculate how many pieces of data are divided into k pieces of data
		
		//例えば 19 = [4,5,5,5]
		//pierre: example 19 = [4,5,5,5]
		vector<uint> tNumDiv;
		for(uint j = 0; j < tK; ++j){
			tNumDiv.push_back((tN + j) / tK);
		}

		#pragma omp parallel
		{
			//各λの正解数をスレッド毎に保持
			//pierre: Keep the number of correct answers for each λ for each thread
			vector<double> tCnt_private(aSolutionPath.size() - 1, 0);
			vector<double> tPrecisions_private(aSolutionPath.size() - 1, 0);
			vector<double> tRecalls_private(aSolutionPath.size() - 1, 0);

			#pragma omp for
			//pierre: for loop for lambda selection
			for(uint i = 0; i < tAve; ++i){
				//index をシャッフル
				//pierre: Shuffle index
				vector<uint> tTrainIdx;
				for(uint j = 0; j < tN; ++j){
					tTrainIdx.push_back(j);
				}

				for(uint j = 0; j < tN; ++j){
					uint k = j + (rand() % (tN - j));
					swap(tTrainIdx[j], tTrainIdx[k]);
				}

				//k-foldの結果がtCntに入る
				//pierre: k-fold results goes in tCnt_private
				if(tLossType==1)
					k_fold_more_metrics(aSolutionPath, aPrefix, aLearner, tTransaction, tY, tK, tCnt_private, tPrecisions_private, 
										tRecalls_private, tTrainIdx, tNumDiv,tLossType);
				else if(tLossType==2)
					k_fold(aSolutionPath, aPrefix, aLearner, tTransaction, tY, tK, tCnt_private, tTrainIdx, tNumDiv,tLossType);
				else std::cout << "error:CV output" << '\n';
			}

			//tCntへ結果をまとめる．念のため排他的に処理
			//pierre: Summarize the results to tCnt. Just to be sure it is exclusively processed
			#pragma omp critical
			//pierre omp critical means that the code is executed one thread at a time
			{
				for(uint j = 0; j < tCnt.size(); ++j){
					tCnt[j] += tCnt_private[j];
				}
			}

		}

		vector<double>::iterator tIter;
		cout << '\n';
		switch (tLossType) {
      // L2SVM
			case 1:
			for(uint i = 0; i < tCnt.size(); ++i){
				cout << "λ[" << i + 1 << "]=" << aSolutionPath[i + 1] 
				<< " Accuracy: " << tCnt[i] << "/" << aOptions[1] * tN << "=" << tCnt[i] / (double) (aOptions[1] * tN) 
				<< " Precision: " << tPrecisions[i]
				<< " Recall: " << tRecalls[i] << '\n';
			}
			//最大値選び
			tIter = max_element(tCnt.begin(), tCnt.end());
			break;
      //Lasso
			case 2:
			for(uint i = 0; i < tCnt.size(); ++i){
				cout << "λ[" << i + 1 << "]=" << aSolutionPath[i + 1] << " RMSE:" <<sqrt(tCnt[i] / (double) (aOptions[1] * tN)) << '\n';
			}
			//最大値選び
			tIter = min_element(tCnt.begin(), tCnt.end());
			break;
			default:
			std::cout << "error:CV output" << '\n';
		}

		return distance(tCnt.begin(), tIter) + 1;
	}

	template<class T> void k_fold(const vector<double> &aSolutionPath, PrefixSpan &aPrefix, T &aLearner, 
								  const vector<vector<Event>> &aTransaction, const vector<double> &aY, const uint &aK, 
								  vector<double> &aCnt, vector<uint> aTrainIdx, vector<uint> aNumDiv,const uint &aLossType){

		vector<uint> tSVMOption = { (uint) aSolutionPath.size() };

		#pragma omp parallel
		{
			//aPrefix, aLearnerはそれぞれのスレッドでprivateに持つようにする
			//pierre: aPrefix, aLearner to be private in each thread
			PrefixSpan tPrefix_private = aPrefix;

			//スレッドごとにインスタンスのコピーを持ちたいので派生クラスTで初期化
			//pierre: I want to have a copy of the instance for each thread, so initialize it with derived class T
			T tLearner_private = aLearner;

			#pragma omp for
			for(uint i = 0; i < aK; ++i){
				vector<uint> tTestIdx;
				vector<uint> tTrainIdx = aTrainIdx;
				uint tSumIdx = 0;
				for(uint j = 0; j < i; ++j){
					tSumIdx += aNumDiv[j];
				}

				//pierre: this allows to only have trainidx has parameter and reconstruct TestIdx
				copy(tTrainIdx.begin() + tSumIdx, tTrainIdx.begin() + tSumIdx + aNumDiv[i], back_inserter(tTestIdx));
				tTrainIdx.erase(tTrainIdx.begin() + tSumIdx, tTrainIdx.begin() + tSumIdx + aNumDiv[i]);

				vector<vector<Event> > tTrainTransaction(tTrainIdx.size());
				vector<double> tTrainY(tTrainIdx.size());
				for(uint j = 0; j < tTrainIdx.size(); ++j){
					tTrainTransaction[j] = aTransaction[tTrainIdx[j]];
					tTrainY[j] = aY[tTrainIdx[j]];
				}

				vector<vector<Event>> tTestTransaction(tTestIdx.size());
				vector<double> tTestY(tTestIdx.size());
				for(uint j = 0; j < tTestIdx.size(); ++j){
					tTestTransaction[j] = aTransaction[tTestIdx[j]];
					tTestY[j] = aY[tTestIdx[j]];
				}

				tPrefix_private.init(tTrainTransaction, tTrainY);

				vector<vector<double> > tYHats = tLearner_private.get_all_predict(tPrefix_private, aSolutionPath, tTestTransaction, tSVMOption);

				calc_accuracy(tYHats, tTestY, aCnt,aLossType);
			}
		}

	}

	//pierre: CUSTOM SOLUTION FOUND, find more elegant way
	template<class T> void k_fold_more_metrics(const vector<double> &aSolutionPath, PrefixSpan &aPrefix, T &aLearner, 
								const vector<vector<Event>> &aTransaction, const vector<double> &aY, const uint &aK, 
								vector<double> &aCnt, vector<double> &aPrecision, vector<double> &aRecall, vector<uint> aTrainIdx, 
								vector<uint> aNumDiv,const uint &aLossType){

	vector<uint> tSVMOption = { (uint) aSolutionPath.size() };

	#pragma omp parallel
	{
		//aPrefix, aLearnerはそれぞれのスレッドでprivateに持つようにする
		//pierre: aPrefix, aLearner to be private in each thread
		PrefixSpan tPrefix_private = aPrefix;

		//スレッドごとにインスタンスのコピーを持ちたいので派生クラスTで初期化
		//pierre: I want to have a copy of the instance for each thread, so initialize it with derived class T
		T tLearner_private = aLearner;

		#pragma omp for
		for(uint i = 0; i < aK; ++i){
			vector<uint> tTestIdx;
			vector<uint> tTrainIdx = aTrainIdx;
			uint tSumIdx = 0;
			for(uint j = 0; j < i; ++j){
				tSumIdx += aNumDiv[j];
			}

			//pierre: this allows to only have trainidx has parameter and reconstruct TestIdx
			copy(tTrainIdx.begin() + tSumIdx, tTrainIdx.begin() + tSumIdx + aNumDiv[i], back_inserter(tTestIdx));
			tTrainIdx.erase(tTrainIdx.begin() + tSumIdx, tTrainIdx.begin() + tSumIdx + aNumDiv[i]);

			vector<vector<Event> > tTrainTransaction(tTrainIdx.size());
			vector<double> tTrainY(tTrainIdx.size());
			for(uint j = 0; j < tTrainIdx.size(); ++j){
				tTrainTransaction[j] = aTransaction[tTrainIdx[j]];
				tTrainY[j] = aY[tTrainIdx[j]];
			}

			vector<vector<Event>> tTestTransaction(tTestIdx.size());
			vector<double> tTestY(tTestIdx.size());
			for(uint j = 0; j < tTestIdx.size(); ++j){
				tTestTransaction[j] = aTransaction[tTestIdx[j]];
				tTestY[j] = aY[tTestIdx[j]];
			}

			tPrefix_private.init(tTrainTransaction, tTrainY);

			vector<vector<double> > tYHats = tLearner_private.get_all_predict(tPrefix_private, aSolutionPath, tTestTransaction, tSVMOption);

			calc_accuracy(tYHats, tTestY, aCnt,aLossType);
			calc_precision(tYHats, tTestY, aPrecision,aLossType);
			calc_recall(tYHats, tTestY, aRecall,aLossType);
		}
	}

}

	void k_fold(const vector<double> &aSolutionPath, PrefixSpan &aPrefix, LearnerSPP &aLearner, const vector<vector<Event> > &aTransaction, const vector<double> &aY, const uint &aK, vector<double> &aCnt,const uint &aLossType);

	//option[k-foldのk,何回の平均をとるか,...]
	uint select(const vector<double> &aSolutionPath, const Database &aDatabase, PrefixSpan &aPrefix, LearnerSPP &aLearner, const vector<uint> &aOptions);
};

#endif
