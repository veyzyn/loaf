export type SkillDefinition = {
  name: string;
  nameLower: string;
  description: string;
  descriptionPreview: string;
  content: string;
  sourcePath: string;
  directoryPath: string;
};

export type SkillCatalog = {
  directory: string;
  directories: string[];
  skills: SkillDefinition[];
  errors: string[];
};

export type SkillSelection = {
  explicitMentions: string[];
  explicit: SkillDefinition[];
  autoMatched: SkillDefinition[];
  combined: SkillDefinition[];
};
