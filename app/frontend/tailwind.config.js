/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        accent: {
          50:  '#f0f4ff',
          100: '#dde6ff',
          200: '#c3d0ff',
          300: '#9fb2ff',
          400: '#7589fc',
          500: '#5162f8',
          600: '#3a46ee',
          700: '#2f38d4',
          800: '#2930ab',
          900: '#272e87',
        },
      },
    },
  },
  plugins: [],
}
