#include "database.h"

void Database::read(const char *aFilename){
  ifstream tFile(aFilename);
  if (!tFile) {
    cerr << "Error: cannot open file" << endl;
  	exit(1);
  }

  uint tEventSize = 0;
  double tLabel;
  string tLine;
  vector<Event> tSequence;

  //全データを読み込む
  while (getline(tFile, tLine)) {
    tSequence.clear();
  	stringstream ss1(tLine);
  	ss1 >> tLabel;
  	mY.push_back(tLabel);
    string eventstring;

    Event tEvent;
  	// ここで１レコードを読む
    while (ss1 >> eventstring) {
      tEvent.clear();
  		stringstream ss2(eventstring);

      string itemstring;
  		int tmp;
  		//ここのループで1:1:1というある時間でのデータを読む
  		while (getline(ss2, itemstring, ':')) {
        tmp = stoi(itemstring);
  			uint tVal = (tmp < 0) ? 0xffffffff : tmp; // wild card
        tEvent.push_back(tVal);
  		}

  		//全要素が同じフォーマットになっているかをチェックしている 1:1:1 と1:1みたいなものは共存できない.
  		if (!tEventSize) {
        tEventSize = tEvent.size();
      } else {
        if (tEventSize != tEvent.size()) {
          cerr << "Format Error: different Event Size at line: " << mTransaction.size() << ", event: " << tSequence.size() << endl;
          exit(-1);
        }
      }

      tSequence.push_back(tEvent);
    }
    mTransaction.push_back(tSequence);
  }
}

vector<vector<Event> > Database::get_transaction()const{
  return mTransaction;
}

vector<double> Database::get_y()const{
  return mY;
}
