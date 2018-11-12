function [err, R, predicted, net] = getErrorBP(data, crossIndex, para, isTrainAtGivenWb)
    %% 计算BP神经网络测试集误差
    %       如果para为空，则用data中的训练集训练后再测试
    %       如果para不为空，则代入para进行测试
    % 输入：
    %   data                - 数据集（算法要求每列是一个样本）
    %   crossIndex          - 当前交叉验证的折
    %   neureCount          
    %   para.neureCount     - 行向量，优化算法推荐的每层神经元个数
    %       .wb             - 列向量，优化算法推荐的BP神经网络参数
    %   isTrainAtGavenWb    - 在设置了wb的条件下是否再使用训练集训练，para为空时无效
    % 输出：
    %   err                 - 算法MSE
    %   R                   - 算法R
    %   predicted           - 预测值（只有一个目标值，行向量）
    %   net                 - 最终预测时用的模型
    
    neuronCount = para.neuronCount;
    perCross = data.crossColumn{crossIndex};
    net = newff(perCross.trainX, perCross.trainY, neuronCount);
    net.trainParam.showWindow = false; 
    
    % 如果para中没有wb，则用data中的训练集训练
    if ~isfield(para, 'wb')
        net = train(net, perCross.trainX, perCross.trainY);
        
    % 如果para中有wb，则代入wb
    else
        wb = para.wb;
        dimX = size(perCross.trainX,1);
        dimY = size(perCross.trainY,1);
        switch size(neuronCount, 2)
            case 1
                % 输入层到隐藏层权重
                for i = 1 : neuronCount
                    inputWeights(i,:) = wb((i - 1) * dimX + 1 : i * dimX, 1)'; 
                end
                net.IW{1,1} = inputWeights;
                % 隐藏层到输出层权重
                for i = 1 : dimY
                    layerWeights(i,:) = wb(dimX * neuronCount + (i - 1) * neuronCount + 1 :...
                        dimX * neuronCount + i * neuronCount, 1)'; 
                end
                net.LW{2,1} = layerWeights;
                % 隐藏层偏置
                for i = 1 : neuronCount
                    hiddenBias(i) = wb(neuronCount * (dimX + dimY) + i, 1);
                end
                net.b{1} = hiddenBias';
                % 输出层偏置
                for i = 1 : dimY
                    outBias(i) = wb(neuronCount(1) * (dimX + dimY + 1) + i, 1);
                end
                net.b{2} = outBias';
            case 2
                % 输入层到隐藏层1权重
                for i = 1 : neuronCount(1)
                    inputWeights(i,:) = wb((i - 1) * dimX + 1 : i * dimX, 1)'; 
                end
                net.IW{1,1} = inputWeights;
                % 隐藏层1到隐藏层2权重
                for i = 1 : neuronCount(2)
                    hiddenWeights(i,:) = wb(dimX * neuronCount(1) + (i - 1) * neuronCount(1) + 1 :...
                        dimX * neuronCount(1) + i * neuronCount(1), 1)';
                end
                net.LW{2,1} = hiddenWeights;
                % 隐藏层到输出层权重
                for i = 1 : dimY
                    layerWeights(i,:) = wb(neuronCount(1) * (neuronCount(2) + dimX) + (i - 1) * neuronCount(2) + 1 :...
                        neuronCount(1) * (neuronCount(2) + dimX) + i * neuronCount(2), 1)'; 
                end
                net.LW{3,2} = layerWeights;
                % 隐藏层1偏置
                for i = 1 : neuronCount(1)
                    hidden1Bias(i) = wb(neuronCount(1) * (neuronCount(2) + dimX) + neuronCount(2) * dimY + i, 1);
                end
                net.b{1} = hidden1Bias';
                % 隐藏层2偏置
                for i = 1 : neuronCount(2)
                    hidden2Bias(i) = wb(neuronCount(1) * (neuronCount(2) + dimX + 1) + neuronCount(2) * dimY + i, 1);
                end
                net.b{2} = hidden2Bias';
                % 输出层偏置
                for i = 1 : dimY
                    outBias(i) = wb(neuronCount(1) * (neuronCount(2) + dimX + 1) + neuronCount(2) * (dimY + 1) + i, 1);
                end
                net.b{3} = outBias';
        end
        if isTrainAtGivenWb == true
            net = train(net, perCross.trainX, perCross.trainY);
        end
    end
    
    predicted = sim(net, perCross.testX);
    err = mse(perCross.testY, predicted);
    R = min(min(corrcoef(perCross.testY, predicted)));