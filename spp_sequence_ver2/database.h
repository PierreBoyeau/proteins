#ifndef DATABASE_H
#define DATABASE_H

#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>

using namespace std;

using uint = unsigned int;
using Event = vector<uint>;

class Database{
private:
  vector<vector<Event> > mTransaction;
  vector<double> mY;
public:
  void read(const char *aFilename);//txtデータをlabelとtransactionの形にする．
  vector<vector<Event> > get_transaction() const; //transactionを取り出す.
  vector<double> get_y() const; //yを取り出す
};

#endif
