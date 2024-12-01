import React from 'react';
import { PieChart, Pie, Cell } from 'recharts';
import AphasiaTypeAnalysis from './AphasiaTypeAnalysis';
import RiskAssessment from './RiskAssessment'; 
import {
  StreamlitComponentBase,
  withStreamlitConnection,
} from "streamlit-component-lib";


interface State {
  count: number;
}

interface CustomLabelProps {
  cx: number;
  cy: number;
  outerRadius: number;
  startAngle: number;
  endAngle: number;
  index: number;
}

interface PatientInfoProps {
  name: string;
  birthYear: number;
  gender: string;
  prediction: string;
  logit_values: {
    Control: number;
    Fluent: number;
    Non_Comprehensive: number;  // 수정
    Non_Fluent: number;  // 수정
  };
  diagnosisDate: string;
}

interface DiagnosisDataItem {
  name: string;
  value: number;
  color: string;
}

class MedicalReport extends StreamlitComponentBase<State> {
  public render = (): React.ReactNode => {
    // Streamlit에서 전달된 데이터 읽기
    const patientInfo: PatientInfoProps = this.props.args["patient_info"] || {
      name: "John Doe",
      birthYear: 2000,
      gender: "male",
      prediction: "Control",
      logit_values: { 
        "Control": 0.6, 
        "Fluent": 0.2, 
        "Non_Comprehensive": 0.1,  // 수정
        "Non_Fluent": 0.1  // 수정
      },
      diagnosisDate: "2024-11-25",
    };

    const logit_values = patientInfo.logit_values || {
      "Control": 0.6,
      "Fluent": 0.2,
      "Non_Comprehensive": 0.1,  // 수정
      "Non_Fluent": 0.1  // 수정
    };

    const diagnosisData: DiagnosisDataItem[] = [
      { 
        name: 'Control', 
        value: patientInfo.logit_values["Control"] || 0, 
        color: '#50D2C2' 
      },
      { 
        name: 'Fluent', 
        value: patientInfo.logit_values["Fluent"] || 0, 
        color: '#4B89FF' 
      },
      { 
        name: 'Non_Comprehensive', 
        value: patientInfo.logit_values["Non_Comprehensive"] || 0, 
        color: '#FF7676' 
      },
      { 
        name: 'Non_Fluent', 
        value: patientInfo.logit_values["Non_Fluent"] || 0, 
        color: '#FFB547' 
      }
    ];

    const renderCustomizedLabel = (props: any) => {
      const { cx, cy, midAngle, innerRadius, outerRadius, value, index } = props;
      const RADIAN = Math.PI / 180;
      const radius = innerRadius + (outerRadius - innerRadius) * 0.5;
      const x = cx + radius * Math.cos(-midAngle * RADIAN);
      const y = cy + radius * Math.sin(-midAngle * RADIAN);
      return (
        <text
          x={x}
          y={y}
          fill="#333"
          textAnchor={x > cx ? 'start' : 'end'}
          dominantBaseline="central"
          fontSize={12}
        >
          {`${diagnosisData[index].name}: ${value}%`}
        </text>
      );
    };

    return (
      <div className="relative max-w-6xl mx-auto min-h-screen bg-white p-8">
        {/* PDF 다운로드 버튼 */}
        <button
          onClick={() => window.print()}
          className="fixed top-4 right-4 z-10 flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors print:hidden"
        >
          PDF Download
        </button>

        {/* 헤더 */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-8">
          <div className="flex justify-between items-start">
            <div>
              <h1 className="text-3xl font-bold text-gray-800">Aphasia Diagnosis Report</h1>
              <p className="text-gray-600 mt-2">Diagnosis Date: {patientInfo.diagnosisDate}</p>
            </div>
            <div className="text-right">
              <p className="text-gray-600">Patient Name: {patientInfo.name}</p>
              <p className="text-gray-600">Gender: {patientInfo.gender}</p>
            </div>
          </div>
        </div>

      {/* 환자 기본 정보 */}
      <div className="bg-white shadow-lg rounded-lg p-6 mb-8">
        <h2 className="text-xl text-blue-600 font-bold mb-4">Patient Information</h2>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="font-semibold text-gray-700">Name</p>
            <p>{patientInfo.name}</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="font-semibold text-gray-700">Birth Year</p>
            <p>{patientInfo.birthYear}년</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="font-semibold text-gray-700">Gender</p>
            <p>{patientInfo.gender}</p>
          </div>
          <div className="bg-gray-50 p-4 rounded-lg">
            <p className="font-semibold text-gray-700">Diagnosis Date</p>
            <p>{patientInfo.diagnosisDate}</p>
          </div>
        </div>
      </div>

      {/* Diagnosis Result Section */}
<div className="bg-white shadow-lg rounded-lg p-6 mb-8">
 <h2 className="text-xl text-blue-600 font-bold mb-4">Diagnosis Result</h2>
 <div className="grid grid-cols-2 gap-4">
   <div className="flex justify-center">
     <PieChart width={400} height={300}>  
       <Pie
         data={diagnosisData}
         cx={150}         
         cy={150}
         innerRadius={45} 
         outerRadius={75}
         label={(props) => {
           const { cx, cy, midAngle, innerRadius, outerRadius, value, index } = props;
           const RADIAN = Math.PI / 180;
           const radius = outerRadius * 1.4;
           const x = cx + radius * Math.cos(-midAngle * RADIAN);
           const y = cy + radius * Math.sin(-midAngle * RADIAN);
           const formattedValue = Number(value).toFixed(2);
           
           return (
             <text
               x={x}
               y={y}
               fill="#333"
               textAnchor={x > cx ? 'start' : 'end'}
               dominantBaseline="central"
               fontSize={12}
             >
               {`${diagnosisData[index].name}: ${value}`}
             </text>
           );
         }}
         dataKey="value"
       >
         {diagnosisData.map((entry, index) => (
           <Cell key={`cell-${index}`} fill={entry.color} />
         ))}
       </Pie>
     </PieChart>
   </div>
   <div className="flex justify-center items-center relative">
     <div className="absolute left-0 h-40 w-px bg-gray-300"></div>
     <div className="text-center">
       {(() => {
         // value를 기준으로 내림차순 정렬
         const sortedData = [...diagnosisData]
           .filter(item => item.name !== 'Control') // Control 제외
           .sort((a, b) => b.value - a.value);

         // 가장 높은 값과 나머지 값들 구하기
         const primary = sortedData[0];
         const secondary = sortedData.slice(1, 3);

         return (
           <>
             <p className="font-bold text-lg mb-2">
               Primary Diagnosis: {primary.name} Aphasia ({(primary.value * 100).toFixed(0)}%)
             </p>
             <p className="text-gray-600">
               Secondary: {secondary.map((item, index) => 
                 `${item.name} (${(item.value * 100).toFixed(0)}%)${index < secondary.length - 1 ? ', ' : ''}`
               )}
             </p>
           </>
         );
       })()}
     </div>
   </div>
 </div>
</div>

      {/* Analysis Section and Risk Assessment Grid */}
      <div className="grid grid-cols-2 gap-8 mb-8">
  <div>
    <AphasiaTypeAnalysis
      logits={{
        Control: diagnosisData[0].value,
        Fluent: diagnosisData[1].value,
        Non_Comprehensive: diagnosisData[2].value,
        Non_Fluent: diagnosisData[3].value
      }}
    />
  </div>
  <RiskAssessment diagnosisData={diagnosisData} />
</div>

      {/* Detailed Examination Section */}
      <div className="bg-white shadow-lg rounded-lg p-6 mb-8">
        <h2 className="text-xl text-blue-600 font-bold mb-4">Detailed Examination</h2>
        <div className="space-y-4">
          {/* Communication Method */}
          <div>
            <p className="font-bold">1. Communication Method</p>
            <div className="flex justify-between items-center mt-2 p-2 bg-gray-50 rounded">
              <p>You: </p>
              <p className="ml-4">~</p>
              <p className="ml-auto">Was it difficult to understand?</p>
            </div>
            <div className="flex justify-between items-center mt-2 p-2 bg-gray-50 rounded">
              <p>Patient: </p>
              <p className="ml-4">~</p>
            </div>
          </div>
          {/* Language Testing */}
          <div className="mt-4">
            <p className="font-bold">2. Language Testing</p>
            <div className="flex justify-between items-center mt-2 p-2 bg-gray-50 rounded">
              <p>You: </p>
              <p className="ml-4">~</p>
            </div>
            <div className="flex justify-between items-center mt-2 p-2 bg-gray-50 rounded">
              <p>Patient: </p>
              <p className="ml-4">~</p>
            </div>
          </div>
        </div>
      </div>

      {/* Recommendations and Conclusion Grid */}
<div className="grid grid-cols-2 gap-8 mb-8">
  <div className="bg-white shadow-lg rounded-lg p-6">
    <h2 className="text-xl text-blue-600 font-bold mb-4">Recommendations</h2>
    <div className="space-y-6">
      {/* Neuroimaging Tests Section */}
      <div className="p-4 bg-blue-50 rounded-lg">
        <h3 className="font-bold text-gray-700 mb-2">Neuroimaging Tests</h3>
        <p className="text-sm text-gray-600 mb-3">
          It is recommended to undergo neuroimaging tests, such as <span className="font-semibold">MRI</span>, 
          to identify potential structural abnormalities in the brain, including stroke or tumor-related causes.
        </p>
        <p className="text-sm text-gray-600 mb-3">
          In cases of suspected degenerative brain diseases, where <span className="font-semibold">MRI</span> may 
          not clearly reveal functional impairments, a <span className="font-semibold">PET scan</span> can be conducted 
          to assess metabolic activity and detect functional deficits in specific brain regions.
        </p>
        <div className="bg-white p-3 rounded-lg">
          <p className="text-sm font-semibold mb-2">Recommendations based on potential abnormalities:</p>
          <div className="ml-4 space-y-1">
            <p className="text-sm">• For structural abnormalities → <span className="font-semibold">MRI</span></p>
            <p className="text-sm">• For functional abnormalities → <span className="font-semibold">PET</span></p>
          </div>
        </div>
      </div>

      {/* Neuropsychological Tests Section */}
      <div className="p-4 bg-blue-50 rounded-lg">
        <h3 className="font-bold text-gray-700 mb-2">Neuropsychological (Cognitive) Tests</h3>
        <p className="text-sm text-gray-600 mb-3">
          Neuropsychological testing is advised to evaluate how the brain processes sensory information, 
          focuses attention, remembers events, thinks logically, produces speech, and performs physical movements.
        </p>
        <p className="text-sm text-gray-600">
          These tests can provide critical insights into cognitive abilities and identify areas requiring 
          further intervention or treatment.
        </p>
      </div>
    </div>
  </div>

  {/* Conclusion Section */}
<div className="bg-white shadow-lg rounded-lg p-6">
 <h2 className="text-xl text-blue-600 font-bold mb-4">Conclusion</h2>
 <div className="p-4 bg-gray-50 rounded-lg">
   {diagnosisData[3].value >= 0.7 ? (
     <p className="text-gray-700">
       Your aphasia rate appears to be somewhat low. You may rest assured! 
       However, if you want a more professional and accurate judgment, you may visit the hospital.
     </p>
   ) : (
     <p className="text-gray-700">
       It seems that the aphasia rate is somewhat high. 
       It is recommended that you visit a nearby hospital for a professional and more accurate diagnosis.
     </p>
   )}
 </div>
</div>
</div>

      {/* Footer */}
      <div className="mt-8 pt-6 border-t-2 border-gray-200">
          <div className="flex justify-between">
            <div>
              <p className="font-semibold text-gray-800">Doctor: Dr. Lee</p>
              <p className="text-sm text-gray-600">License No: 12345</p>
            </div>
            <div className="text-right">
              <p className="font-semibold text-gray-800">Aphasia Diagnostic Center</p>
              <p className="text-sm text-gray-600">contact@adc.com</p>
            </div>
          </div>
        </div>
      </div>
    );
  }
}

export default withStreamlitConnection(MedicalReport);