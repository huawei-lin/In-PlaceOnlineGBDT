mex CXXFLAGS="$CXXFLAGS -fopenmp -DOMP -O3" LDFLAGS="$LDFLAGS -fopenmp -DOMP -O3" onlineGBDT_train.cpp ../data.cpp ../tree.cpp ../model.cpp -I..
mex CXXFLAGS="$CXXFLAGS -fopenmp -DOMP -O3" LDFLAGS="$LDFLAGS -fopenmp -DOMP -O3" onlineGBDT_test.cpp ../data.cpp ../tree.cpp ../model.cpp -I.. 
mex CXXFLAGS="$CXXFLAGS -fopenmp -DOMP -O3" LDFLAGS="$LDFLAGS -fopenmp -DOMP -O3" onlineGBDT_load.cpp ../data.cpp ../tree.cpp ../model.cpp -I..
mex CXXFLAGS="$CXXFLAGS -fopenmp -DOMP -O3" LDFLAGS="$LDFLAGS -fopenmp -DOMP -O3" onlineGBDT_save.cpp ../data.cpp ../tree.cpp ../model.cpp -I..
mex CXXFLAGS="$CXXFLAGS -fopenmp -DOMP -O3" LDFLAGS="$LDFLAGS -fopenmp -DOMP -O3" onlineGBDT_predict.cpp ../data.cpp ../tree.cpp ../model.cpp -I..
mex libsvmread.c

