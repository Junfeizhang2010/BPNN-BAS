function analysisSensitivity(data, model, tuneMode, algoMode, opt)
    %% 敏感性分析
    % 输入：
    %   data    - 数据
    %   model   - 训练好的模型
    
    disp(['            ●正在分析数据指标对' algoMode '模型的敏感性...']);
    all = data.all;
    %% 计算分值，公式：△y/y
    n = data.inputDimCount;
    for i = 1 : n
        l = min(all(:,i));
        h = max(all(:,i));
        unit{i} = linspace(l, h * 0.9, 10)';
    end  
    x1 = eachBlock(n, unit);  % 每列一个样本
    switch algoMode
        case 'BP'
            predicted1 = sim(model, x1');  % 行向量
        case 'SVM'
            predicted1 = svmpredict(rand(size(x1, 1), 1), x1, model);  % 算法要求必须输入真实的结果，用于计算MSE，此处不需要，随机数输入即可
        case 'LR'
            predicted1 = 1 ./ (1 + exp([ones(size(x1, 1), 1), -x1] * model));
        otherwise
            predicted1 = predict(model, x1);
    end
    for i = 1 : n
        x2 = [x1(:,i) * 1.1, x1(:, [1 : i - 1 ,i + 1 : end])];
        switch algoMode
            case 'BP'
                predicted2 = sim(model, x2');  % 行向量
            case 'SVM'
                predicted2 = svmpredict(rand(size(x2, 1), 1), x2, model);
            case 'LR'
                predicted2 = 1 ./ (1 + exp([ones(size(x2, 1), 1), -x2] * model));
            otherwise
                predicted2 = predict(model, x2);
        end
        score(i) = mean(mean((predicted2 - predicted1) ./ predicted1)); % 打分公式
    end
    [~, index] = sort(abs(score), 'descend');
    temp = score(index); % 此处必须用temp，因为下面的画图函数要求score不能被排序，否则与数据中的标签不对应
    figurePara = setFigurePara(data.dataIndex, data.posOutput);
    fprintf(['              数据指标对' algoMode '模型的敏感性分析完毕，指标敏感性分数依次为：']);
    for i = 1 : size(temp, 2)
        fprintf([figurePara.label.input{index(i)}, ' — ' ,num2str(temp(i))]);
        if i ~= size(temp, 2)
            fprintf('，');
        else
            disp('。');
        end
    end
    
    %% 画图
    if ~isempty(opt) || ~isfield(opt, 'show') || ~isfield(opt.show, 'isDraw') || opt.show.isDraw
        figurePara = setFigurePara(data.dataIndex, data.posOutput);
        figurePara.label.title = ['Sensitivity Analysis Tornato Graph of Influential Variable with ' algoMode];
        figurePara.fold = ['savedFigure\' data.dataIndex '\' algoMode '\' tuneMode];
        figurePara.preserve.name = ['使用' tuneMode '算法优化的' algoMode '模型指标敏感性分析龙卷风图'];
        figurePara.label.y = 'Importance score';
        figurePara.label.x = 'Influential variable';
        drawTornado(score, figurePara); % score不可提前排序，否则与数据标签不对应
    end
    
function block = eachBlock(i, unit)
    if i == 1
        block = unit{1};
    else
        block = eachBlock(i - 1, unit);
        block = repmat(block, 10, 1);
        block = cat(2, block, reshape(repmat(unit{i}', 10^(i-1), 1), 10^i, 1));
    end