// Dataframe Class Header
// Ben Crabtree, 2021

# ifndef __DATAFRAME_HPP
# define __DATAFRAME_HPP

# include <string>
# include <fstream>
# include <utility> // std::pair,
# include <vector>
# include <stdexcept>
# include <sstream>
# include <map>

class Dataframe
{
    int Rows;
    int Cols;
    bool Header;
    std::vector<std::string> ColNames;
    std::vector<std::vector<float>> Vals;

    public:
    
    Dataframe();
    ~Dataframe();

    // Reads a csv file into an empty dataframe. Assumes csv contains numerical (float) data.
    // csv may also contain a column of class labels (strings) which will be encoded to floats
    // based on the label_map provided.
    void read_csv(std::string filename, bool header, std::map<std::string, float> label_map = {{"", 0}});

    void setColNames(std::vector<std::string> names);
    std::vector<std::string> getColNames();
    
    void setHeader(bool header);
    bool getHeader();

    void setVals(std::vector<std::vector<float>> vals);
    std::vector<std::vector<float>> getVals();
    
    void setNumRows(int numRows);
    int getNumRows();
    
    void setNumCols(int numCols);
    int getNumCols();
    
    // Prints first numRows rows
    void head(int numRows);
    // Prints last numRows rows
    void tail(int numRows);
    // Prints whole dataframe
    void printDf();
    
    // Returns a train dataframe and test dataframe,
    // splitting original dataframe based on percentages specified.
    std::pair<Dataframe, Dataframe> trainTestSplit(int trainPercent, int testPercent);
    // Returns vector of numFolds many dataframes, each of equal length
    // (apart from last fold, which may be longer if there are rows left over)
    std::vector<Dataframe> crossValSplit(int numFolds);

    // Combines two dataframes by column (side by side)
    void colBind(Dataframe df);
    // Combines two dataframes by row (one on top of other)
    void rowBind(Dataframe df);

    Dataframe operator[](std::vector<std::string> names); // Access columns by name
    Dataframe operator[](std::vector<int> indices); // Access rows by index

};

# endif