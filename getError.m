function err = getError(data, crossIndex, algoMode, x, others)
    %%  �����㷨MSE
    % ���룺
    %   data                - ���ݼ�
    %   algrMode            - �㷨ģʽ�� BPStructure���Ż�BP������ṹ
    %                                   BPParameter���Ż�BP���������
    %
    %   x                   - ���������㷨����������������ŵĲ�����
    %   others              - ������������� BPStructure     ~
    %                                       BPParameter     others.neuronCount   �Ż�1������ʱ����Ԫ����
    % �����
    %   err                 - �㷨MSE

    switch algoMode
        case 'BPStructure'
            para.neuronCount = x';
        case 'BPParameter'
            para.neuronCount = others.neuronCount;
            para.wb = x;
    end
    n = 1;  % �ظ�ʵ�����
    for j = 1 : n
        switch algoMode
            case {'BPStructure', 'BPParameter'}
                err(j) = getErrorBP(data, crossIndex, para, false);
            otherwise
                err(j) = getErrorNormal(data, crossIndex, x, algoMode);
        end
    end
    err = mean(err);