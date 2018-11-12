function untunedCompare(data, tunedErr, tunedR, tuneMode, algoMode, opt)
    %% 比较未经优化情况
     % 输入：
    %   data      - 数据集
    %   tunedErr  - 十折交叉验证的测试集误差均值
    %   tunedR    - 十折交叉验证的测试集相关系数均值
    %   tuneMode  - 优化算法
    %   algoMode  - 模型算法
    %   opt       - 算法参数
    
    %% 计算随机情况，共（折数*10）次
    disp(['            ●正在验证经优化的' algoMode '模型与未经优化的差异...']);
    fprintf('                  ');
    time = size(data.crossColumn, 2);
    if isfield(opt, 'compare') && isfield(opt.compare, 'pointCount')
        multiple = floor(opt.compare.pointCount / time);
    else
        multiple = 10;
    end
    
    for j = 1 : time
        for i = 0 : multiple
            if i * 10 + j > opt.compare.pointCount, j = time; break; end
            switch algoMode
                case 'BP'
                    para.neuronCount = ceil(rand(1,ceil(rand(1) * 4)) * 60);
                    [err(i * 10 + j), R(i * 10 + j)] = getErrorBP(data, j, para, []);
                case 'SVM'
                    [err(i * 10 + j), R(i * 10 + j), predicted] =...
                            getErrorNormal(data, j, rands(1, 2) * 7 - 3, algoMode);
                case {'DT', 'RF'}
                    [err(i * 10 + j), R(i * 10 + j), predicted] =...
                            getErrorNormal(data, j, round(rand(1, 2) * 100 + 1), algoMode);
                case 'KNN'
                    [err(i * 10 + j), R(i * 10 + j), predicted] =...
                            getErrorNormal(data, j, round(rand(1) * 50 + 1), algoMode);
                case 'LR'
                    [err(i * 10 + j), R(i * 10 + j), predicted] =...
                            getErrorNormal(data, j, rand(data.inputDimCount + 1, 1) * 5, algoMode);
            end
        end
        fprintf(['已完成' num2str(10 * j) '次测试']);
        if j ~= time
            fprintf('；');
            if rem(j, 5) == 0, fprintf('\n                  '); end
        else
            fprintf('。\n');
        end
    end
    disp(['              验证经优化的' algoMode '模型与未经优化的差异完毕，优化的'...
        algoMode '模型超过' num2str(length(err(err >= tunedErr)) / length(err) * 100) '%的未优化模型。']);
    
    %% 画图
    if ~isempty(opt) ||~isfield(opt, 'show') || ~isfield(opt.show, 'isDraw') || opt.show.isDraw
        % MSE优化对比图
        figurePara = setFigurePara(data.dataIndex, data.posOutput);
        figurePara.label.title = ['Tune MSE and untune MSE of TrainSet Predictive with' algoMode];
        figurePara.fold = ['savedFigure\' data.dataIndex '\' algoMode '\' tuneMode];
        figurePara.preserve.name = ['使用' tuneMode '算法优化的' algoMode '模型优化效果误差对比图'];
        figurePara.label.y = figurePara.label.mse;
        drawTunedAndUntuned(tunedErr, err, figurePara);
        
        % R优化对比图
        figurePara = setFigurePara(data.dataIndex, data.posOutput);
        figurePara.label.title = ['Tune MSE and untune MSE of TrainSet Predictive with ' algoMode];
        figurePara.fold = ['savedFigure\' data.dataIndex '\' algoMode '\' tuneMode];
        figurePara.preserve.name = ['使用' tuneMode '算法优化的' algoMode '模型优化效果相关系数对比图'];
        figurePara.label.y = 'R';
        drawTunedAndUntuned(tunedR, R, figurePara);
    end