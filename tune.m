function [para, bestIteration] = tune(data, tuneMode, algoMode, opt)
    %% �Ż�ȷ��ģ�Ͳ���
    % ���룺
    %   data            - ���ݼ�
    %   tuneMode        - �Ż��㷨
    %   algoMode        - ģ���㷨
    %   opt             - �㷨����
    % �����
    %   para            - �Ż���Ĳ���
    %   bestIteration   - ��������ͼ����
    
    switch algoMode
        case 'BP'
            % �ṹ�Ż�ȷ��
            [neuronCount, bestIteration] = tuneBPStructure(data, tuneMode, opt); % ����data��Ϊ����ʮ�۽�����֤ȡ����ֵ
%             neuronCount = 11; % ������
%             neuronCount = [5 2];
            if length(neuronCount) > 2, neuronCount = 11; end
            % ������ʼֵ�Ż�ȷ��
%             wb = tuneBPParameters(data, neuronCount, tuneMode, opt);
            para.neuronCount = neuronCount;
%             para.wb = wb;
        otherwise
            [para, bestIteration] = tuneNormal(data, tuneMode, algoMode, opt);
    end