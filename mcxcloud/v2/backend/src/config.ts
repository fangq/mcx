export interface Config {
  port: number;
  host: string;
  databaseUrl: string;
  corsOrigin: string;
  threshold: number;
  workerSecret: string;
}

export const config: Config = {
  port: Number(process.env.PORT ?? 8080),
  host: process.env.HOST ?? '0.0.0.0',
  databaseUrl: process.env.DATABASE_URL ?? 'postgres://mcxcloud@localhost/mcxcloud',
  corsOrigin: process.env.CORS_ORIGIN ?? 'https://mcx.space',
  threshold: Number(process.env.BLOB_THRESHOLD ?? 4096),
  workerSecret: process.env.WORKER_SECRET ?? 'change-me',
};
