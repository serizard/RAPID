import React from 'react';

// 기능 상태를 위한 타입 정의
type FunctionalityState = 'normal' | 'impaired';

// 기능 설명을 위한 인터페이스
interface Functionalities {
  fluency: {
    normal: string;
    impaired: string;
  };
  comprehension: {
    normal: string;
    impaired: string;
  };
  repetition: {
    normal: string;
    impaired: string;
  };
  naming: {
    normal: string;
    impaired: string;
  };
}

// 평가 결과를 위한 인터페이스
interface EvaluationResult {
  fluency: FunctionalityState;
  comprehension: FunctionalityState;
  repetition: FunctionalityState;
  naming: FunctionalityState;
}

// 실어증 타입 정의
type AphasiaType = 'Control' | 'Fluent' | 'Non_Comprehensive' | 'Non_Fluent';

// Props 인터페이스
interface AphasiaTypeAnalysisProps {
  logits: Record<AphasiaType, number>;
}

const AphasiaTypeAnalysis: React.FC<AphasiaTypeAnalysisProps> = ({ logits }) => {
  // 정상 및 손상된 기능 정의
  const functionalities: Functionalities = {
    fluency: {
      normal: "Speech is fluent and grammatically correct, with coherent content. There is no frequent use of filler words or nonsensical phrases.",
      impaired: "Speech may be fluent but lacks meaning or coherence. Frequent use of filler words or nonsensical phrases may occur in fluent aphasia types.",
    },
    comprehension: {
      normal: "Able to understand both simple, everyday conversations and complex instructions or abstract language with ease.",
      impaired: "Demonstrates difficulty with complex instructions or abstract language, though simple conversations may still be understood.",
    },
    repetition: {
      normal: "Able to accurately repeat sentences or words without errors, regardless of their length or complexity.",
      impaired: "May struggle to accurately repeat sentences or words, depending on the aphasia type. Fluent aphasia often results in repetition errors despite fluent speech.",
    },
    naming: {
      normal: "Words, particularly nouns or verbs, are recalled effortlessly without hesitation or substitutions.",
      impaired: "Difficulty recalling specific words, particularly nouns or verbs. Responses may include circumlocution (describing the word instead of naming it) or unrelated substitutions.",
    },
  };

  // 실어증 타입에 대한 평가 결과 매핑
  const aphasiaTypes: Record<AphasiaType, EvaluationResult> = {
    Control: {
      fluency: "normal",
      comprehension: "normal",
      repetition: "normal",
      naming: "impaired",
    },
    Fluent: {
      fluency: "impaired",
      comprehension: "normal",
      repetition: "impaired",
      naming: "impaired",
    },
    Non_Comprehensive: {
      fluency: "normal",
      comprehension: "impaired",
      repetition: "impaired",
      naming: "impaired",
    },
    Non_Fluent: {
      fluency: "normal",
      comprehension: "normal",
      repetition: "normal",
      naming: "normal"
    }
  };

  // 타입 안전한 방식으로 최상위 진단 가져오기
  const entries = (Object.entries(logits) as [AphasiaType, number][]);
  const topDiagnosis = entries.reduce((prev, curr) => 
  logits[prev[0]] * 100 > logits[curr[0]] * 100 ? prev : curr
)[0] as AphasiaType;

  // 최상위 진단에 대한 평가 결과 가져오기
  const evaluationResults = aphasiaTypes[topDiagnosis];

  return (
    <div className="bg-white shadow-lg rounded-lg p-6">
      <h2 className="text-xl text-blue-600 font-bold mb-4">Aphasia Type Analysis</h2>
      <p className="text-gray-600 mb-4">Top Diagnosis: {topDiagnosis}</p>
      <div className="space-y-4">
        {/* Fluency */}
        <div className="p-4 bg-blue-50 rounded-lg">
          <h4 className="font-bold">Fluency</h4>
          <p className="text-sm text-gray-600 mt-2">
            {functionalities.fluency[evaluationResults.fluency]}
          </p>
        </div>

        {/* Auditory Comprehension */}
        <div className="p-4 bg-blue-50 rounded-lg">
          <h4 className="font-bold">Auditory Comprehension</h4>
          <p className="text-sm text-gray-600 mt-2">
            {functionalities.comprehension[evaluationResults.comprehension]}
          </p>
        </div>

        {/* Repetition Ability */}
        <div className="p-4 bg-blue-50 rounded-lg">
          <h4 className="font-bold">Repetition Ability</h4>
          <p className="text-sm text-gray-600 mt-2">
            {functionalities.repetition[evaluationResults.repetition]}
          </p>
        </div>

        {/* Naming Ability */}
        <div className="p-4 bg-blue-50 rounded-lg">
          <h4 className="font-bold">Naming Ability</h4>
          <p className="text-sm text-gray-600 mt-2">
            {functionalities.naming[evaluationResults.naming]}
          </p>
        </div>
      </div>
    </div>
  );
};

export default AphasiaTypeAnalysis;