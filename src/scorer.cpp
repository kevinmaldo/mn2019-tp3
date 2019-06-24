#include <iostream>
#include <sstream>
#include <unordered_map>
#include <map>
#include <vector>
#include <algorithm>

using namespace std;

map<string, int> header;

bool isHeader(string& line) {
	return line[0] == 'Y';
}

map<string, int> parseHeader(string line) {
	map<string, int> res;
	
	stringstream ss(line);
	
	string cur;
	while(getline(ss, cur, ',')) {
		int nextIdx = res.size();
		res[cur] = nextIdx;
	}
	
	return res;
}

stringstream ss;
pair<string, double> parseLine(string& line, int groupNumber) {
	ss.clear();
	ss << line;
	
	string key;
	int depTime, CRSDepTime, arrTime, CRSArrTime;
	int idx[] = {header["DepTime"], header["CRSDepTime"], header["ArrTime"], header["CRSArrTime"]};
	
	string cur; int i = 0;
	while(getline(ss, cur, ',')) {
		if (i == groupNumber) key = cur;
		if (idx[0] == i) depTime = (cur == "NA"? -1 : stoi(cur));
		if (idx[1] == i) CRSDepTime = (cur == "NA"? -1 : stoi(cur));
		if (idx[2] == i) arrTime = (cur == "NA"? -1 : stoi(cur));
		if (idx[3] == i) CRSArrTime = (cur == "NA"? -1 : stoi(cur));
		
		i++;
	}
	
	return {key, (depTime == -1 || CRSDepTime == -1 || arrTime == -1 || CRSArrTime == -1)? 15 : min(30, max(abs(depTime - CRSDepTime), abs(arrTime - CRSArrTime)))};
}

bool reported(vector<string>& parsedLine, string column) {
	return parsedLine[header[column]] != "NA";
}

int getIntValue(vector<string>& parsedLine, string column) {
	return stoi(parsedLine[header[column]]);
}

double getScore(vector<string>& parsedLine) {
	if (!reported(parsedLine, "DepTime") || !reported(parsedLine, "CRSDepTime") || !reported(parsedLine, "ArrTime") || !reported(parsedLine, "CRSArrTime")) return 15;
	return min(30, max(abs(getIntValue(parsedLine, "DepTime") - getIntValue(parsedLine, "CRSDepTime")), abs(getIntValue(parsedLine, "ArrTime") - getIntValue(parsedLine, "CRSArrTime"))));
}

int main(int argc, char** argv) {
	cin.tie(NULL);
	ios::sync_with_stdio(false);
	
	string toScore = argv[1];
	unordered_map<string, double> sumOfScores;
	unordered_map<string, int> scoresConsidered;
	
	string line;
	while (getline(cin, line)) {
		if (isHeader(line)) {
			header = parseHeader(line);
		} else {
			pair<string, double> parsed = parseLine(line, header[toScore]);
			
			sumOfScores[parsed.first] += parsed.second;		
			scoresConsidered[parsed.first]++;
		}
	}
	
	vector<pair<double, string>> res;
	for (auto& airlineScores : sumOfScores) {
		res.push_back({airlineScores.second / scoresConsidered[airlineScores.first], airlineScores.first});
	}
	
	sort(res.begin(), res.end());
	
	for (auto& airlineScore : res) {
		cout << airlineScore.second << "," << airlineScore.first << "," << scoresConsidered[airlineScore.second] << '\n';
	}
}
