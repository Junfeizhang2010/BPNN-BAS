function err = getError(data, crossIndex, algoMode, x, others)
    %%  计算算法MSE
    % 输入：
    %   data                - 数据集
    %   algrMode            - 算法模式， BPStructure：优化BP神经网络结构
    %                                   BPParameter：优化BP神经网络参数
    %
    %   x                   - 列向量，算法所需参数（即被调优的参数）
    %   others              - 其他所需参数， BPStructure     ~
    %                                       BPParameter     others.neuronCount   优化1层网络时的神经元个数
    % 输出：
    %   err                 - 算法MSE

    switch algoMode
        case 'BPStructure'
            para.neuronCount = x';
        case 'BPParameter'
            para.neuronCount = others.neuronCount;
            para.wb = x;
    end
    n = 1;  % 重复实验册数
    for j = 1 : n
        switch algoMode
            case {'BPStructure', 'BPParameter'}
                err(j) = getErrorBP(data, crossIndex, para, false);
            otherwise
                err(j) = getErrorNormal(data, crossIndex, x, algoMode);
        end
    end
    err = mean(err);