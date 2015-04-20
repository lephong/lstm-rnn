################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/AbstractNN.cpp \
../src/AbstractParam.cpp \
../src/AbstractTrainer.cpp \
../src/Classifier.cpp \
../src/Dictionary.cpp \
../src/Matrix.cpp \
../src/RNN.cpp \
../src/RNNParam.cpp \
../src/RNNTrainer.cpp \
../src/SenBinTree.cpp \
../src/Treebank.cpp \
../src/Utils.cpp \
../src/main.cpp 

OBJS += \
./src/AbstractNN.o \
./src/AbstractParam.o \
./src/AbstractTrainer.o \
./src/Classifier.o \
./src/Dictionary.o \
./src/Matrix.o \
./src/RNN.o \
./src/RNNParam.o \
./src/RNNTrainer.o \
./src/SenBinTree.o \
./src/Treebank.o \
./src/Utils.o \
./src/main.o 

CPP_DEPS += \
./src/AbstractNN.d \
./src/AbstractParam.d \
./src/AbstractTrainer.d \
./src/Classifier.d \
./src/Dictionary.d \
./src/Matrix.d \
./src/RNN.d \
./src/RNNParam.d \
./src/RNNTrainer.d \
./src/SenBinTree.d \
./src/Treebank.d \
./src/Utils.d \
./src/main.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -I/opt/OpenBLAS/include -O3 -Wall -c -fmessage-length=0 -std=c++11 -fopenmp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


