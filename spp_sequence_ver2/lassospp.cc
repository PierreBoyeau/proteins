#include "lassospp.h"

double LASSOSPP::get_lambda_max(PrefixSpan &aPrefix){

  mN = aPrefix.mY.size();
  mBias = 0;

  double tSum = 0;
  for (uint i = 0; i < mN; ++i) {
    tSum += aPrefix.mY[i];
  }

  //bias項のみで学習
  tSum /= mN;

  mR.shrink_to_fit();
  mR.resize(mN);

  for (uint i = 0; i < mN; ++i) {
    //損失
    mR[i] = aPrefix.mY[i] - tSum;
  }

  aPrefix.get_maxnode(mR,2);

  return aPrefix.Maxnode.val;
}

//sppは木構造を利用しているのでデータではなくaPrefixを入力する
void LASSOSPP::learn(PrefixSpan &aPrefix,const vector<double> &aLambdas,const vector<uint> &aOptions){

  if(aOptions.empty()){
    cout << "error:lasso.learn option size is incorrect." << '\n';
    exit(1);
  }
  //λを求めるための処理
  if(aLambdas.empty()){
    cout << "error:lambda is empty. Set lambda." << '\n';
    exit(1);
  }
  mT = aLambdas.size();
  double tTrueLambdaMax = get_lambda_max(aPrefix);

  // compute solution path
  uint activesize = 1;
  double L1norm = 0;
  double maxval = tTrueLambdaMax; //lambda_max
  for (uint t = 1; (t < aOptions[0])&&(t < mT); ++t) {

		double lam = aLambdas[t];

    if(lam> tTrueLambdaMax){
      cout << "skip:λ="<<lam<<" > λmax="<<tTrueLambdaMax << '\n';
      continue;
    }

  	vector<uint> index;
  	vector<uint *> x;
  	vector<double *> support;
  	// vector<uint> norm;
  	vector<double> norm;
  	vector<uint> xSize;
  	vector<double> grad;
  	vector<double> w;
  	for (uint iter = 0; iter <= mMaxIter; ++iter) {

  		// calculate dual and safe screening
  		if (iter % mFreq == 0) {

  			double loss = 0;
  			double oTr = 0;
  			for (uint i = 0; i < mN; ++i) {
  					loss += mR[i]*mR[i];
  					oTr += aPrefix.mY[i]*mR[i];
  			}

				if (iter >= 1) {
					maxval = 0;
					for (uint s = 0; s < activesize; ++s) {
						uint j = index[s];
						grad[j] = 0;
  					// for (uint k = 0; k < norm[j]; ++k) {
  					for (uint k = 0; k < xSize[j]; ++k) {
  						uint id = x[j][k];
  						grad[j] += mR[id]*support[j][k];
  					}
  					if (fabs(grad[j]) > maxval) maxval = fabs(grad[j]);
  				}
  			}

  			double alpha = min(max(oTr/(lam*loss), -1/maxval), 1/maxval);
  			double primal = 0.5*loss+lam*L1norm;
  			double dual = -0.5*lam*lam*alpha*alpha*loss + lam*alpha*oTr;
  			double gap = primal-dual;

  			if (gap/primal < mEps) {
          if(index.empty()){
            break;
          }
  				uint active = 0;
  				for (uint s = 0; s < activesize; s++) {
  					uint j = index[s];
  					aPrefix.Active[j]->w = w[j];
  					if (w[j] != 0) active++;
  				}

  				// cout << "[iter " << iter << "] primal: " << primal << ", dual: " << dual << ", gap: " << gap/primal << ", activeset: " << active << endl;
  				break;
  			}

  			double radius = sqrt(2*gap)/lam;
  			if (iter == 0) {
  				//各λループのはじめに一回だけscreeningを行う(現在のラムダでの最大の木を構築している)
  				//λは徐々に小さくなって行くの木は前回のλより恐らく大きくなる
  				//w更新するたびバウンドがきつくなり，ある周期で非アクティブを決め，木が小さくなって行く
  				aPrefix.safe_screening(mR, alpha, radius,2);
  				activesize = aPrefix.Active.size();

  				index.resize(activesize);
  				x.resize(activesize);
  				norm.resize(activesize);
  				grad.resize(activesize);
  				w.resize(activesize);
  				xSize.resize(activesize);
  				support.resize(activesize);

  				for (uint j = 0; j < activesize; ++j) {
  					index[j] = j;
  					x[j] = aPrefix.Active[j]->x.data();
  					support[j] = aPrefix.Active[j]->support.data();
  					// norm[j] = aPrefix.Active[j]->x.size();
  					xSize[j] = aPrefix.Active[j]->x.size();
  					norm[j] = aPrefix.Active[j]->supportSum;
						grad[j] = 0;
  					w[j] = aPrefix.Active[j]->w;
  				}

  			} else {
  				//最初の１回目以外は非アクティブノード検出を行う．（木の大きさは変わらない）
  				for (uint s = 0; s < activesize; ++s) {
  					uint j = index[s];
  					double score = fabs(alpha*grad[j])+radius*sqrt(norm[j]);

  					if (score < 1) {
  						if (w[j] != 0) {
  							// for (uint k = 0; k < norm[j]; ++k) {
  							for (uint k = 0; k < xSize[j]; ++k) {
  								uint id = x[j][k];
  								mR[id] += w[j]*support[j][k];
  							}
  						}
  						//学習中にwが0になってもActiveのサイズは変更していない
  						aPrefix.Active[j]->w = 0;
  						activesize--;
  						swap(index[s], index[activesize]);
  						s--;
  					}
  				}
  			}
  			// cout << "[iter " << iter << "] primal: " << primal << ", dual: " << dual << ", gap: " << gap/primal << ", activeset: " << activesize << endl;
  		}

      //ランダムに混ぜている
  		for (uint j = 0; j < activesize; ++j) {
  			uint i = j+rand()%(activesize-j);
  			swap(index[i], index[j]);
  		}

      // update w
      //shooting algorithm
      L1norm = 0;

      for (uint s = 0; s < activesize; ++s) {
        uint j = index[s];
        double xTr = 0;
        //頻度も考えているため shooting algorithm 時にサポートの二乗和が必要
        double tSqrtSupSum = 0;

        for (uint i = 0; i < xSize[j]; ++i) {
          tSqrtSupSum += support[j][i]*support[j][i];
          uint id = x[j][i];
          mR[id] += w[j]*support[j][i];
          xTr += mR[id]*support[j][i];
        }

        if (xTr > lam) {
          w[j] = (xTr-lam)/tSqrtSupSum;
        } else if (xTr < -lam) {
          w[j] = (xTr+lam)/tSqrtSupSum;
        } else {
          w[j] = 0;
        }

        for (uint i = 0; i < xSize[j]; ++i) {
          uint id = x[j][i];
          mR[id] -= w[j]*support[j][i];
        }

        L1norm += fabs(w[j]);
      }

      //bias
      double tmp = 0;
      for (uint i = 0; i < mN; ++i) {
        mR[i] += mBias;
        tmp += mR[i];
      }
      mBias = tmp/mN;
      for (uint i = 0; i < mN; ++i) {
        mR[i] -= mBias;
      }
  	}

  	aPrefix.add_history(t);
  }

}

vector<double> LASSOSPP::predict(const PrefixSpan &aPrefix,const vector<vector<Event> > &aTransaction) const{
  uint tTransactionSize = aTransaction.size();
  vector<double> tYHat(tTransactionSize,mBias);//要素を全てBiasで初期化
  for(auto tActiveNode :aPrefix.Active){
    vector<Event> tPattern = tActiveNode->pattern;
    double tW = tActiveNode->w;
    for(uint i=0;i<tTransactionSize;++i){
      tYHat[i] += tW*aPrefix.calcSup(aTransaction[i],tPattern);
    }
  }

  return tYHat;
}

vector<vector<double> > LASSOSPP::get_all_predict(PrefixSpan &aPrefix,const vector<double> &aLambdas,const vector<vector<Event> > &aTransaction,const vector<uint> &aOptions){

  if(aOptions.empty()){
    cout << "error:lasso.learn option size is incorrect." << '\n';
    exit(1);
  }
  //λを求めるための処理
  if(aLambdas.empty()){
    cout << "error:lambda is empty. Set lambda." << '\n';
    exit(1);
  }
  mT = aLambdas.size();
  double tTrueLambdaMax = get_lambda_max(aPrefix);

  //とりあえずsolution path すべてで学習を行う
  vector<vector<double> > tYHats;
  

  // compute solution path
  uint activesize = 1;
  double L1norm = 0;
  double maxval = tTrueLambdaMax; //lambda_max
  for (uint t = 1; (t < aOptions[0])&&(t < mT); ++t) {
  double lam = aLambdas[t];

    if(lam>tTrueLambdaMax){
      cout << "skip:λ="<<lam<<" > λmax="<<tTrueLambdaMax << '\n';
      vector<double> tV(aTransaction.size(),mBias);
      tYHats.push_back(tV);
      continue;
    }

    vector<uint> index;
    vector<uint *> x;
    vector<double *> support;
    // vector<uint> norm;
    vector<double> norm;
    vector<uint> xSize;
    vector<double> grad;
    vector<double> w;
    for (uint iter = 0; iter <= mMaxIter; ++iter) {

      // calculate dual and safe screening
      if (iter % mFreq == 0) {
        
        double loss = 0;
        double oTr = 0;
        for (uint i = 0; i < mN; ++i) {
            loss += mR[i]*mR[i];
            oTr += aPrefix.mY[i]*mR[i];
        }
        if (iter >= 1) {
          maxval = 0;
          for (uint s = 0; s < activesize; ++s) {

            uint j = index[s];
            grad[j] = 0;

            // for (uint k = 0; k < norm[j]; ++k) {
            for (uint k = 0; k < xSize[j]; ++k) {
              uint id = x[j][k];
              grad[j] += mR[id]*support[j][k];
              //@
            }
            if (fabs(grad[j]) > maxval) maxval = fabs(grad[j]);
          }
        }

        double alpha = min(max(oTr/(lam*loss), -1/maxval), 1/maxval);
        double primal = 0.5*loss+lam*L1norm;
        double dual = -0.5*lam*lam*alpha*alpha*loss + lam*alpha*oTr;
        double gap = primal-dual;
        if (gap/primal < mEps) {
          //最初から収束条件を満たしている時
          if(index.empty()){
            break;
          }
          uint active = 0;
          for (uint s = 0; s < activesize; s++) {
            uint j = index[s];
            aPrefix.Active[j]->w = w[j];
            if (w[j] != 0) active++;
          }

          cout << "λ:"<< t <<"[iter " << iter << "] primal: " << primal << ", dual: " << dual << ", gap: " << gap/primal << ", activeset: " << active << endl;
          break;
        }
        double radius = sqrt(2*gap)/lam;
        if (iter == 0) {

          //各λループのはじめに一回だけscreeningを行う(現在のラムダでの最大の木を構築している)
          //λは徐々に小さくなって行くの木は前回のλより恐らく大きくなる
          //w更新するたびバウンドがきつくなり，ある周期で非アクティブを決め，木が小さくなって行く
          aPrefix.safe_screening(mR, alpha, radius,2);
          activesize = aPrefix.Active.size();

          index.resize(activesize);
          x.resize(activesize);
          norm.resize(activesize);
          grad.resize(activesize);
          w.resize(activesize);
          xSize.resize(activesize);
          support.resize(activesize);

          for (uint j = 0; j < activesize; ++j) {
            index[j] = j;
            x[j] = aPrefix.Active[j]->x.data();
            support[j] = aPrefix.Active[j]->support.data();
            // norm[j] = aPrefix.Active[j]->x.size();
            xSize[j] = aPrefix.Active[j]->x.size();
            norm[j] = aPrefix.Active[j]->supportSum;
            grad[j] = 0;
            w[j] = aPrefix.Active[j]->w;
          }

        } else {
          //最初の１回目以外は非アクティブノード検出を行う．（木の大きさは変わらない）
          for (uint s = 0; s < activesize; ++s) {
            uint j = index[s];
            double score = fabs(alpha*grad[j])+radius*sqrt(norm[j]);

            if (score < 1) {
              if (w[j] != 0) {
                // for (uint k = 0; k < norm[j]; ++k) {
                for (uint k = 0; k < xSize[j]; ++k) {
                  uint id = x[j][k];
                  mR[id] += w[j]*support[j][k];
                }
              }
              //学習中にwが0になってもActiveのサイズは変更していない
              aPrefix.Active[j]->w = 0;
              activesize--;
              swap(index[s], index[activesize]);
              s--;
            }
          }
        }
        //  cout << "[iter " << iter << "] primal: " << primal << ", dual: " << dual << ", gap: " << gap/primal << ", activeset: " << activesize << endl;
      }
      //ランダムに混ぜている
      for (uint j = 0; j < activesize; ++j) {
        uint i = j+rand()%(activesize-j);
        swap(index[i], index[j]);
      }


      // update w
      //shooting algorithm
      L1norm = 0;

      for (uint s = 0; s < activesize; ++s) {
        uint j = index[s];
        double xTr = 0;
        //頻度も考えているため shooting algorithm 時にサポートの二乗和が必要
        double tSqrtSupSum = 0;

        for (uint i = 0; i < xSize[j]; ++i) {
          tSqrtSupSum += support[j][i]*support[j][i];
          uint id = x[j][i];
          mR[id] += w[j]*support[j][i];
          xTr += mR[id]*support[j][i];
        }

        if (xTr > lam) {
          w[j] = (xTr-lam)/tSqrtSupSum;
        } else if (xTr < -lam) {
          w[j] = (xTr+lam)/tSqrtSupSum;
        } else {
          w[j] = 0;
        }

        for (uint i = 0; i < xSize[j]; ++i) {
          uint id = x[j][i];
          mR[id] -= w[j]*support[j][i];
        }

        L1norm += fabs(w[j]);
      }


      //bias
      double tmp = 0;
      for (uint i = 0; i < mN; ++i) {
        mR[i] += mBias;
        tmp += mR[i];
      }
      mBias = tmp/mN;
      for (uint i = 0; i < mN; ++i) {
        mR[i] -= mBias;
      }

    }

    tYHats.push_back(predict(aPrefix,aTransaction));


  }

  return tYHats;
}
