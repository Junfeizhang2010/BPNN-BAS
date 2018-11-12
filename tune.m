function [para, bestIteration] = tune(data, tuneMode, algoMode, opt)
    %% 优化确认模型参数
    % 输入：
    %   data            - 数据集
    %   tuneMode        - 优化算法
    %   algoMode        - 模型算法
    %   opt             - 算法参数
    % 输出：
    %   para            - 优化后的参数
    %   bestIteration   - 迭代收敛图数据
    
    switch algoMode
        case 'BP'
            % 结构优化确定
            [neuronCount, bestIteration] = tuneBPStructure(data, tuneMode, opt); % 传入data是为了用十折交叉验证取最优值
%             neuronCount = 11; % 测试用
%             neuronCount = [5 2];
            if length(neuronCount) > 2, neuronCount = 11; end
            % 参数初始值优化确定
%             wb = tuneBPParameters(data, neuronCount, tuneMode, opt);
            para.neuronCount = neuronCount;
%             para.wb = wb;
        otherwise
            [para, bestIteration] = tuneNormal(data, tuneMode, algoMode, opt);
    end