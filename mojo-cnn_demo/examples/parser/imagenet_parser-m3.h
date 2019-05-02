#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <iostream> // cout
#include <sstream>
#include <fstream>
#include <iomanip> //setw
#include <random>


using namespace cv;
using namespace std;

namespace imagenet
{
	std::string data_name() {return std::string("IMAGENET");}

	bool parse_test_data(std::string &data_path, std::vector<std::vector<float>> &test_records, std::vector<int> &test_labels, 
			float min_val=-1.f, float max_val=1.f, int x_padding=0, int y_padding=0)
	{
		// x_padding = 0;
		// y_padding = 0;
		ifstream label_val(data_path+"val.txt");

		vector<pair<string, int> > lines;
		string line;
		size_t pos;
		int label;
		while (getline(label_val, line)) 
		{
			pos = line.find_last_of(' ');
			label = atoi(line.substr(pos + 1).c_str());
			lines.push_back(make_pair(line.substr(0, pos), label));
		}
		//random_shuffle(lines.begin(), lines.end());
		for (int line_id = 0; line_id < lines.size(); ++line_id) 
		{
			string filename = lines[line_id].first;
			int label = lines[line_id].second;

			string recordpath = data_path+"val/"+filename;

			/* new parser core */
			std::ifstream infile_feat(recordpath); //加载数据文件
			std::string feature; //存储读取的每行数据
			float feat_onePoint;  //存储每行按空格分开的每一个float数据
			std::vector<float> lines; //存储每行数据
			//存储所有数据
			test_records.clear();

			while(!infile_feat.eof()) 
			{	
				getline(infile_feat, feature); //一次读取一行数据
				stringstream stringin(feature); //使用串流实现对string的输入输出操作
				lines.clear();
				while (stringin >> feat_onePoint) {      //按空格一次读取一个数据存入feat_onePoint 
					lines.push_back(feat_onePoint); //存储每行按空格分开的数据 
				}
				test_records.push_back(lines); //存储所有数据
			}
			infile_feat.close();
			test_labels.push_back(label);
		}
		label_val.close();
		return true;
	}

	bool parse_train_data(std::string &data_path, std::vector<std::vector<float>> &train_records, std::vector<int> &train_labels, 
			float min_val=-1.f, float max_val=1.f, int x_padding=0, int y_padding=0)
	{
		//x_padding = 0;
		//y_padding = 0;
		ifstream label_train(data_path+"/train.txt");

		vector<pair<string, int> > lines;
		string line;
		size_t pos;
		int label;
		while (getline(label_train, line)) 
		{
			pos = line.find_last_of(' ');
			label = atoi(line.substr(pos + 1).c_str());
			lines.push_back(make_pair(line.substr(0, pos), label));
		}

		random_shuffle(lines.begin(), lines.end());

		for (int line_id = 0; line_id < lines.size(); ++line_id) 
		{
			string filename = lines[line_id].first;
			int label = lines[line_id].second;

			//string imagepath = data_path+"/train/"+filename;
			string recordpath = data_path+"train/"+filename;

			/* new parser core */
			std::ifstream infile_feat(recordpath); //加载数据文件
			/* Weijie: test */
			cout << recordpath << endl;

			std::string feature; //存储读取的每行数据
			float feat_onePoint;  //存储每行按空格分开的每一个float数据
			std::vector<float> lines; //存储每行数据
			//存储所有数据
			train_records.clear();

			while(!infile_feat.eof()) 
			{	
				getline(infile_feat, feature); //一次读取一行数据
				stringstream stringin(feature); //使用串流实现对string的输入输出操作
				lines.clear();
				while (stringin >> feat_onePoint) {      //按空格一次读取一个数据存入feat_onePoint 
					/* Weijie: test */
					lines.push_back(feat_onePoint); //存储每行按空格分开的数据 
				}
				train_records.push_back(lines); //存储所有数据
			}
			infile_feat.close();



			train_labels.push_back(label);
		}

		cout<<"finished parsing training dataset"<<endl;
		label_train.close();
		return true;
	}
}
