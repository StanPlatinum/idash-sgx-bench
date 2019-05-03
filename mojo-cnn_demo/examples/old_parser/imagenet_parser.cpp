#include <vector>
#include <iostream>
#include "imagenet_parser.h"

using namespace std;
using namespace imagenet;


std::string data_path="./testdata/";

int main(int argc, char *argv[])
{
    //int width, height;

    vector<vector<float>> train_records;
    vector<int> train_labels;

    if (!parse_train_data(data_path, train_records, train_labels)) {
	 std::cerr << "error: could not parse the sample data.\n"; 
	return 1; 
    }

    cout << "=================" << endl;
    ofstream out("output.txt");
    

    /* deal with train_labels */
    cout << "reading labels...";
    for (int i = 0; i < train_labels.size(); ++i)
    {
	cout << i << endl;
    }
    /* deal with train_records */
    vector<float> tmp_vector;

    if (out.is_open()) 
    {
        out << "This is a line.\n";
        out << "This is another line.\n";
    }

    cout << "reading records..." << endl;
    cout << "writting records..." << endl;
    //for (vector<vector<float>>::iterator ite = train_records.begin(); ite != train_records.end(); ++ite)
    for (int i = 0; i < train_records.size(); ++i)
    {
	tmp_vector = train_records[i];
        cout << "writing vector row " << i << " ..." << endl;
	out << "row" << i << ": ";
	for (vector<float>::iterator itee = tmp_vector.begin(); itee != tmp_vector.end(); itee++){
            //cout << "writing record: ";
	    out << *itee << " ";
	}
	out << endl;
    }
    out.close();

#if 0
    Mat img = imread("150013000229.jpg", IMREAD_COLOR);
    Mat cropped_img(227, 227, CV_8UC3);
    
    cv::resize(img, cropped_img, cv::Size(227, 227));
	   
    namedWindow( "test",  WINDOW_AUTOSIZE);
    imshow("test",  cropped_img);
    waitKey(0);
#endif
    return 0;
}
