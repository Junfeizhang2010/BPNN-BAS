function [err, R, predicted, net] = getErrorBP(data, crossIndex, para, isTrainAtGivenWb)
    %% ����BP��������Լ����
    %       ���paraΪ�գ�����data�е�ѵ����ѵ�����ٲ���
    %       ���para��Ϊ�գ������para���в���
    % ���룺
    %   data                - ���ݼ����㷨Ҫ��ÿ����һ��������
    %   crossIndex          - ��ǰ������֤����
    %   neureCount          
    %   para.neureCount     - ���������Ż��㷨�Ƽ���ÿ����Ԫ����
    %       .wb             - ���������Ż��㷨�Ƽ���BP���������
    %   isTrainAtGavenWb    - ��������wb���������Ƿ���ʹ��ѵ����ѵ����paraΪ��ʱ��Ч
    % �����
    %   err                 - �㷨MSE
    %   R                   - �㷨R
    %   predicted           - Ԥ��ֵ��ֻ��һ��Ŀ��ֵ����������
    %   net                 - ����Ԥ��ʱ�õ�ģ��
    
    neuronCount = para.neuronCount;
    perCross = data.crossColumn{crossIndex};
    net = newff(perCross.trainX, perCross.trainY, neuronCount);
    net.trainParam.showWindow = false; 
    
    % ���para��û��wb������data�е�ѵ����ѵ��
    if ~isfield(para, 'wb')
        net = train(net, perCross.trainX, perCross.trainY);
        
    % ���para����wb�������wb
    else
        wb = para.wb;
        dimX = size(perCross.trainX,1);
        dimY = size(perCross.trainY,1);
        switch size(neuronCount, 2)
            case 1
                % ����㵽���ز�Ȩ��
                for i = 1 : neuronCount
                    inputWeights(i,:) = wb((i - 1) * dimX + 1 : i * dimX, 1)'; 
                end
                net.IW{1,1} = inputWeights;
                % ���ز㵽�����Ȩ��
                for i = 1 : dimY
                    layerWeights(i,:) = wb(dimX * neuronCount + (i - 1) * neuronCount + 1 :...
                        dimX * neuronCount + i * neuronCount, 1)'; 
                end
                net.LW{2,1} = layerWeights;
                % ���ز�ƫ��
                for i = 1 : neuronCount
                    hiddenBias(i) = wb(neuronCount * (dimX + dimY) + i, 1);
                end
                net.b{1} = hiddenBias';
                % �����ƫ��
                for i = 1 : dimY
                    outBias(i) = wb(neuronCount(1) * (dimX + dimY + 1) + i, 1);
                end
                net.b{2} = outBias';
            case 2
                % ����㵽���ز�1Ȩ��
                for i = 1 : neuronCount(1)
                    inputWeights(i,:) = wb((i - 1) * dimX + 1 : i * dimX, 1)'; 
                end
                net.IW{1,1} = inputWeights;
                % ���ز�1�����ز�2Ȩ��
                for i = 1 : neuronCount(2)
                    hiddenWeights(i,:) = wb(dimX * neuronCount(1) + (i - 1) * neuronCount(1) + 1 :...
                        dimX * neuronCount(1) + i * neuronCount(1), 1)';
                end
                net.LW{2,1} = hiddenWeights;
                % ���ز㵽�����Ȩ��
                for i = 1 : dimY
                    layerWeights(i,:) = wb(neuronCount(1) * (neuronCount(2) + dimX) + (i - 1) * neuronCount(2) + 1 :...
                        neuronCount(1) * (neuronCount(2) + dimX) + i * neuronCount(2), 1)'; 
                end
                net.LW{3,2} = layerWeights;
                % ���ز�1ƫ��
                for i = 1 : neuronCount(1)
                    hidden1Bias(i) = wb(neuronCount(1) * (neuronCount(2) + dimX) + neuronCount(2) * dimY + i, 1);
                end
                net.b{1} = hidden1Bias';
                % ���ز�2ƫ��
                for i = 1 : neuronCount(2)
                    hidden2Bias(i) = wb(neuronCount(1) * (neuronCount(2) + dimX + 1) + neuronCount(2) * dimY + i, 1);
                end
                net.b{2} = hidden2Bias';
                % �����ƫ��
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