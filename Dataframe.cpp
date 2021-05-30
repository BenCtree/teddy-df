// Dataframe Class Implementation
// Ben Crabtree, 2021

# include "Dataframe.hpp"
# include <iostream>
# include <string>
# include <fstream>
# include <utility> // std::pair,
# include <vector>
# include <stdexcept>
# include <sstream>
# include <algorithm>
# include <map>
# include <random>
# include <algorithm>
# include <iterator>
# include <math.h>

Dataframe::Dataframe()
{
    Rows = 0;
    Cols = 0;
    Header = false;
    ColNames = {};
    Vals = {};
}

Dataframe::~Dataframe()
{
    // Destructor
}

// Helper function to encode class labels as floats
float conversion(std::string elem, std::map<std::string, float> label_map)
{
    for (auto const& pair : label_map)
    {
        std::string key = pair.first;
        float value = pair.second;
        // If elem is in label_map, return its encoded value
        if (elem == key)
        {
            return value;
        }
    }
    // If elem is not in lable map, it is not a class label - convert it to a float
    return std::stof(elem);
}

void Dataframe::read_csv(std::string filename, bool header, std::map<std::string, float> label_map)
{
    // Column names (1st row of csv if header == true)
    std::vector<std::string> names;
    // Each following row is a vector of floats followed by a string (class label)
    std::vector<std::vector<float>> values;

    std::ifstream file(filename);
    if(!file.is_open())
    {
        throw std::runtime_error("Error: Problem opening file.");
    }

    std::string line;
    std::string colname;

    if (file.good())
    {
        if (header == true)
        {
            Header = true;
            // Get first line (header) of csv file
            std::getline(file, line);
            // Convert line to stringstream so we can read from it with getline()
            std::stringstream ss(line);

            // Read each name into names vector
            while (std::getline(ss, colname, ','))
            {
                names.push_back(colname);
            }
        }
        ColNames = names;
        
        //int rowidx = 0;
        std::string elem;
    
        while(std::getline(file, line))
        {
            std::vector<float> rowdata;
            std::stringstream ss(line);
            float converted;

            while(std::getline(ss, elem, ','))
            {
                converted = conversion(elem, label_map);
                rowdata.push_back(converted);
            }
            values.push_back(rowdata);
        }
        // Set Vals to point at values
    }
    file.close();
    Vals = values;
    Rows = (int)values.size();
    Cols = (int)values[0].size();
}

void Dataframe::setColNames(std::vector<std::string> names)
{
    ColNames = names;
}

std::vector<std::string> Dataframe::getColNames()
{
    return ColNames;
}

bool Dataframe::getHeader()
{
    return Header;
}

void Dataframe::setHeader(bool header)
{
    Header = header;
}

void Dataframe::setVals(std::vector<std::vector<float>> vals)
{
    Vals = vals;
}

std::vector<std::vector<float>> Dataframe::getVals()
{
    return Vals;
}

void Dataframe::setNumRows(int numRows)
{
    Rows = numRows;
}

int Dataframe::getNumRows()
{
    return Rows;
}

void Dataframe::setNumCols(int numCols)
{
    Cols = numCols;
}

int Dataframe::getNumCols()
{
    return Cols;
}

void Dataframe::head(int numRows)
{
    // Print colnames
    if (Header == true)
    {
        for (int j = 0; j < Cols; j++)
        {
            std::cout << ColNames[j] << " ";
        }
        std::cout << std::endl;
    }
    // Print values
    for (int i = 0; i < numRows; i++)
    {
        for (int j = 0; j < Cols; j++)
        {
            std::cout << Vals[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void Dataframe::tail(int numRows)
{
    // Print colnames
    if (Header == true)
    {
        for (int j = 0; j < Cols; j++)
        {
            std::cout << ColNames[j] << " ";
        }
        std::cout << std::endl;
    }
    // Print values
    int start = Rows - numRows;
    for (int i = start; i < Rows; i++)
    {
        for (int j = 0; j < Cols; j++)
        {
            std::cout << Vals[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void Dataframe::printDf()
{
    // Print colnames
    if (Header == true)
    {
        for (int j = 0; j < Cols; j++)
        {
            std::cout << ColNames[j] << " ";
        }
        std::cout << std::endl;
    }
    // Print values
    for (int i = 0; i < Rows; i++)
    {
        for (int j = 0; j < Cols; j++)
        {
            std::cout << Vals[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

std::pair<Dataframe, Dataframe> Dataframe::trainTestSplit(int trainPercent, int testPercent)
{
    if (trainPercent + testPercent != 100)
    {
        std::cerr << "Error: Percentages must add up to 100.";
        throw "Incorrect percentage sum error.";
    }
    std::vector<int> rowIndices;
    for (int i = 0; i < Rows; i++)
    {
        rowIndices.push_back(i);
    }
    
    std::random_device rd; // random number from hardware
    std::mt19937 g(rd()); // seed generator
 
    std::shuffle(rowIndices.begin(), rowIndices.end(), g); // permute rowIndices

    int num_train_rows = round(((float)Rows / 100.0) * trainPercent);
    int num_test_rows = Rows - num_train_rows;

    std::vector<std::vector<float>> train_rows;
    for (int i = 0; i < num_train_rows; i++)
    {
        train_rows.push_back(Vals[rowIndices[i]]);
    }
    
    std::vector<std::vector<float>> test_rows;
    for (int i = num_train_rows; i < Rows; i++)
    {
        test_rows.push_back(Vals[rowIndices[i]]);
    }
    
    Dataframe trainDf = Dataframe();
    trainDf.setColNames(getColNames());
    trainDf.setHeader(true);
    trainDf.setVals(train_rows);
    trainDf.setNumCols(Cols);
    trainDf.setNumRows(num_train_rows);
    
    Dataframe testDf = Dataframe();
    testDf.setColNames(getColNames());
    testDf.setHeader(true);
    testDf.setVals(test_rows);
    testDf.setNumCols(Cols);
    testDf.setNumRows(num_test_rows);

    return std::make_pair(trainDf, testDf);
}

std::vector<Dataframe> Dataframe::crossValSplit(int numFolds)
{
    int num_rows = floor((float)Rows / (float)numFolds);
    int rem = Rows - (num_rows * numFolds);
    std::vector<Dataframe> folds;
    for (int i = 0; i < numFolds; i++)
    {
        Dataframe fold = Dataframe();
        std::vector<std::vector<float>> rows;
        for (int j = (i*num_rows); j < ((i+1)*num_rows); j++)
        {
            rows.push_back(Vals[j]);
        }
        // If there are rows left over, add them to last fold
        fold.setNumRows(num_rows);
        if (rem != 0 && i == numFolds-1)
        {
            int rem_idx = Rows - rem;
            for (int k = rem_idx; k < Rows; k++)
            {
                rows.push_back(Vals[k]);
                fold.setNumRows(num_rows + rem);
            }
        }
        fold.setColNames(getColNames());
        fold.setHeader(true);
        fold.setVals(rows);
        fold.setNumCols(Cols);
        folds.push_back(fold);
    }
    return folds;
}

// Helper function to get col index from col name
int getIndex(std::vector<std::string> v, std::string s)
{
    int index;
    auto it = std::find(v.begin(), v.end(), s);
    if (it != v.end()) 
    {
        index = std::distance(v.begin(), it);
        return index;
    }
    else
    {
        return -1;
    }
}

void Dataframe::colBind(Dataframe df)
{
    std::vector<std::string> df_names = df.getColNames();
    for (std::string name : df_names)
    {
        ColNames.push_back(name);
    }
    std::vector<std::vector<float>> df_vals = df.getVals();
    // If dataframe being added to is empty
    if (Rows == 0)
    {
        for (int i = 0; i < df.getNumRows(); i++)
        {
            std::vector<float> row;
            for (int j = 0; j < df.getNumCols(); j++)
            {
                row.push_back(df_vals[i][j]);
            }
            Vals.push_back(row);
        }
        setNumRows(Rows + df.getNumRows());
        if (Cols == 0)
        {
            setNumCols(Cols + df.getNumCols());
        }
    }
    // If dataframe being added to is not empty
    else
    {
        for (int i = 0; i < df.getNumRows(); i++)
        {
            for (int j = 0; j < df.getNumCols(); j++)
            {
                Vals[i].push_back(df_vals[i][j]);
            }
        }
        setNumCols(Cols + df.getNumCols());
    }
    if (df.getHeader())
    {
        setHeader(true);
    }
}

void Dataframe::rowBind(Dataframe df)
{
    if (Cols == 0)
    {
        setColNames(df.getColNames());
        if (df.getHeader())
        {
            setHeader(true);
        }
    }
    std::vector<std::vector<float>> df_vals = df.getVals();
    for (std::vector<float> row : df_vals)
    {
        Vals.push_back(row);
    }
    setNumRows(Rows + df.getNumRows());
    if (Cols == 0)
    {
        setNumCols(df.getNumCols());
    }
}

Dataframe Dataframe::operator[](std::vector<std::string> names)
{
    Dataframe colSubset = Dataframe();
    for (std::string name : names)
    {
        int colidx = getIndex(ColNames, name);
        if (colidx >= 0 && colidx < Cols)
        {
            Dataframe col = Dataframe();
            col.setColNames(std::vector<std::string>{name});
            col.setHeader(true);
            std::vector<std::vector<float>> vals = {};
            for (int i = 0; i < Rows; i++)
            {
                std::vector<float> rowval = {};
                rowval.push_back(Vals[i][colidx]);
                vals.push_back(rowval);
            }
                col.setVals(vals);
                col.setNumRows(Rows);
                col.setNumCols(1);
                colSubset.colBind(col);
                if (col.getHeader())
                {
                    colSubset.setHeader(true);
                }
        }
        else
        {
            std::cerr << "Error: Col name provided is not in ColNames.";
            throw "Index out of range error.";
        }
    }
    return colSubset;
}

Dataframe Dataframe::operator[](std::vector<int> indices)
{
    Dataframe rowSubset = Dataframe();
    for (int i : indices)
    {
        if (i >= 0 && i < Rows)
        {
            Dataframe row = Dataframe();
            row.setColNames(ColNames);
            row.setHeader(true);
            row.setVals(std::vector<std::vector<float>>{Vals[i]});
            row.setNumRows(1);
            row.setNumCols(Cols);
            rowSubset.rowBind(row);
            if (row.getHeader())
            {
                rowSubset.setHeader(true);
            }
        }
        else
        {
            std::cerr << "Error: Index must be within range.";
            throw "Index out of range error.";
        }
    }
    return rowSubset;
}