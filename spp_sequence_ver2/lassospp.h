#ifndef LASSOSPP_H
#define LASSOSPP_H

#include "prefixspan.h"
#include "database.h"
#include "learnerspp.h"
#include <vector>

using namespace std;

using uint = unsigned int;
using Event = vector<uint>;

//Lasso
class LASSOSPP : public LearnerSPP{
private:

  uint mN;

  double mBias;
  vector<double> mR;
  uint mT;
  uint mMaxIter;
  uint mFreq;
  double mEps;
  double mRatio;
  double clac_sup(const vector<Event> &aSequence,const vector<Event> &aPattern,const uint aSupMode);

public:
  LASSOSPP(uint aMaxIter, uint aFreq, double aEps,double aRatio){
  mMaxIter = aMaxIter;
  mFreq = aFreq;
  mEps = aEps;
  mRatio = aRatio;
  };
  //aOption[0]に1~aLambdas.size()までの数字をi入れることでaLambdas[i-1]での学習を行う
  virtual void learn(PrefixSpan &aPrefix,const vector<double> &aLambdas,const vector<uint> &aOptions);
  //λmaxの値を計算して返す
  virtual double get_lambda_max(PrefixSpan &aPrefix);
  //現在学習されているモデルで予測を行う
  virtual vector<double> predict(const PrefixSpan &aPrefix,const vector<vector<Event> > &aTransaction) const;
  //solution path全てででの予測値Y^を返す. aOption[0]の設定learnと同じ
  virtual vector<vector<double> > get_all_predict(PrefixSpan &aPrefix,const vector<double> &aLambdas,const vector<vector<Event> > &aTransaction,const vector<uint> &aOptions);
};

#endif
